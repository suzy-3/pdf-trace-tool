from typing import Literal
from pathlib import Path
import secrets
import threading
import time

import numpy as np
import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .processing import clean, png, to_svg
from .cad import CadLimitError, cad_clean

app = FastAPI(title='PDFトレースツール')
STATIC = Path(__file__).parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC), name='static')
# Single-worker, temporary in-memory storage. Expired documents are removed on access.
TTL = 3600
MAX_BYTES = 20 * 1024 * 1024
MAX_DOCUMENTS = 8
MAX_PIXELS = 12_000_000
store = {}
lock = threading.RLock()


class Options(BaseModel):
    mode: Literal['legacy', 'cad_clean'] = 'legacy'
    line_width: float = Field(default=0.5, ge=0.1, le=1.5)
    noise_strength: float = Field(default=0, ge=0, le=3)
    join_threshold: float = Field(default=0.1, ge=0, le=1)
    snap: bool = False
    fidelity: Literal['faithful', 'clean'] = 'faithful'
    deskew: bool = True
    denoise: bool = True
    sharpen: bool = True
    min_area: int = Field(default=6, ge=1, le=200)


def document(token):
    now = time.monotonic()
    for key in list(store):
        if now - store[key]['created'] > TTL:
            del store[key]
    if token not in store:
        raise HTTPException(404, '図面が見つかりません。再アップロードしてください。')
    return store[token]


def page_image(item, page):
    with pymupdf.open(stream=item['pdf'], filetype='pdf') as doc:
        if page < 1 or page > len(doc):
            raise HTTPException(404, 'ページが存在しません。')
        pdf_page = doc[page - 1]
        rect = pdf_page.rect
        scale = min(2.0, (MAX_PIXELS / max(1, rect.width * rect.height)) ** 0.5)
        pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                  colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.post('/api/documents')
async def upload(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'PDFは20MB以下にしてください。')
    try:
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if doc.needs_pass or not 1 <= len(doc) <= 100:
                raise ValueError()
            pages = len(doc)
    except Exception:
        raise HTTPException(400, '有効なPDFを選択してください（暗号化なし・1〜100ページ）。')
    with lock:
        now = time.monotonic()
        for key in list(store):
            if now - store[key]['created'] > TTL:
                del store[key]
        if len(store) >= MAX_DOCUMENTS:
            raise HTTPException(503, '一時保存が満杯です。不要な図面を閉じてください。')
        token = secrets.token_urlsafe(24)
        store[token] = dict(pdf=data, pages=pages, created=now, results={})
    return {'id': token, 'pages': pages}


@app.delete('/api/documents/{token}')
def delete(token: str):
    with lock:
        document(token)
        del store[token]
    return {'deleted': True}


@app.get('/api/documents/{token}/pages/{page}/original.png')
def original(token: str, page: int):
    with lock:
        result = png(page_image(document(token), page))
    return Response(result, media_type='image/png')


# One heavy operation at a time; progress never waits on the image-work lock.
processing_slot = threading.BoundedSemaphore(1)
BINARY_FIELDS = {'png', 'svg', 'reference', 'difference', 'page_size', '_stored_at'}
RESULT_CACHE_BYTES = 128 * 1024 * 1024


def summary(output):
    return {k: v for k, v in output.items() if k not in BINARY_FIELDS}


def publish(item, page, output):
    """Called under lock. Bound retained output buffers across all pages/docs."""
    def weight(value):
        return sum(len(v) for v in value.values() if isinstance(v, bytes))
    required = weight(output)
    if required > RESULT_CACHE_BYTES:
        raise CadLimitError('出力の保存容量が128MBを超えます。ページを分割してください。')
    entries = [(value.get('_stored_at', 0), results, key, weight(value))
               for document_item in store.values()
               for results in [document_item['results']] for key, value in results.items()]
    used = sum(entry[3] for entry in entries)
    for _, results, key, size in sorted(entries, key=lambda entry: entry[0]):
        if used + required <= RESULT_CACHE_BYTES:
            break
        del results[key]
        used -= size
    output['_stored_at'] = time.monotonic()
    item['results'][page] = output


def compute(item, page, options, progress=None):
    if progress:
        progress('ページを読み込み', 0, 0, 1)
    source = page_image(item, page)
    if options.mode == 'cad_clean':
        with pymupdf.open(stream=item['pdf'], filetype='pdf') as doc:
            rect = doc[page-1].rect
        scale = min(2.0, (MAX_PIXELS/max(1, rect.width*rect.height))**0.5)
        output = cad_clean(source, deskew=options.deskew, scale=scale, progress=progress,
                           **{k: getattr(options, k) for k in
                              ('line_width', 'noise_strength', 'join_threshold', 'snap', 'fidelity')})
        output['mode'] = 'cad_clean'
        output['page_size'] = (output['width']/scale, output['height']/scale)
        return output
    image, angle = clean(source, deskew=options.deskew, denoise=options.denoise,
                         sharpen=options.sharpen, min_area=options.min_area)
    return dict(png=png(image), svg=to_svg(image), angle=round(angle, 2),
                width=image.shape[1], height=image.shape[0])


def processing_error(exc):
    if isinstance(exc, HTTPException):
        return exc.status_code, str(exc.detail)
    if isinstance(exc, CadLimitError):
        return 413, str(exc)
    if isinstance(exc, MemoryError):
        return 503, '処理用メモリが不足しました。他の図面を閉じて再実行してください。'
    if isinstance(exc, OSError):
        return 503, '一時ファイルの保存に失敗しました。ディスク容量または保存先の権限を確認してください。'
    return 500, '図面の処理に失敗しました。PDFが正常に開けることを確認し、設定を変更して再実行してください。'


@app.post('/api/documents/{token}/pages/{page}/process')
def process(token: str, page: int, options: Options):
    if not processing_slot.acquire(blocking=False):
        raise HTTPException(409, '別の図面を処理中です。完了後に再実行してください。')
    try:
        with lock:
            item = document(token)
            item['results'].pop(page, None)
        output = compute(item, page, options)
        with lock:
            if store.get(token) is item:
                publish(item, page, output)
        return summary(output)
    except Exception as exc:
        code, message = processing_error(exc)
        raise HTTPException(code, message) from exc
    finally:
        processing_slot.release()


@app.post('/api/documents/{token}/pages/{page}/jobs', status_code=202)
def start_job(token: str, page: int, options: Options):
    if not processing_slot.acquire(blocking=False):
        raise HTTPException(409, '別の図面を処理中です。完了後に再実行してください。')
    try:
        with lock:
            item = document(token)
            if not 1 <= page <= item['pages']:
                raise HTTPException(404, 'ページが存在しません。')
            job_id = secrets.token_urlsafe(24)
            job = dict(id=job_id, page=page, state='running', stage='開始待ち',
                       percent=0, completed=0, total=1)
            # Keep only the latest job per document, not an unbounded history.
            item['job'] = job
            item['results'].pop(page, None)
        def report(stage, percent, completed, total):
            with lock:
                if store.get(token) is not item:
                    raise HTTPException(404, '図面が削除されたため処理を中止しました。')
                job.update(stage=stage, percent=max(job['percent'], min(99, int(percent))),
                           completed=completed, total=total)
        def run():
            try:
                output = compute(item, page, options, report)
                with lock:
                    if store.get(token) is item:
                        publish(item, page, output)
                        job.update(state='done', percent=100, stage='完了', result=summary(output))
            except Exception as exc:
                code, message = processing_error(exc)
                with lock:
                    job.update(state='failed', stage='処理失敗', error=message, error_code=code)
            finally:
                processing_slot.release()
        threading.Thread(target=run, name='pdf-cleanup', daemon=True).start()
        return dict(job)
    except Exception as exc:
        processing_slot.release()
        code, message = processing_error(exc)
        with lock:
            if 'job' in locals():
                job.update(state='failed', stage='処理失敗', error=message, error_code=code)
        raise HTTPException(code, message) from exc


@app.get('/api/documents/{token}/pages/{page}/jobs/{job_id}')
def job_status(token: str, page: int, job_id: str):
    with lock:
        job = document(token).get('job')
        if not job or job['id'] != job_id or job['page'] != page:
            raise HTTPException(404, '処理状況が見つかりません。再変換してください。')
        return dict(job)


@app.get('/api/documents/{token}/pages/{page}/{kind}')
def result(token: str, page: int, kind: str):
    if kind not in ('corrected.png', 'trace.svg', 'corrected.pdf', 'reference.png', 'difference.png'):
        raise HTTPException(404, '出力形式が存在しません。')
    with lock:
        item = document(token)
        if page not in item['results']:
            raise HTTPException(409, '先にこのページを補正してください。')
        output = item['results'][page]
        if kind in ('reference.png', 'difference.png'):
            key = kind.split('.')[0]
            if key not in output:
                raise HTTPException(409, '重ね合わせ・差分はCAD清書モードで実行してください。')
            return Response(output[key], media_type='image/png', headers={'Cache-Control': 'no-store'})
        if kind == 'corrected.pdf' and output.get('mode') == 'cad_clean':
            with pymupdf.open(stream=output['svg'], filetype='svg') as drawing:
                vector_pdf = drawing.convert_to_pdf()
            with pymupdf.open(stream=vector_pdf, filetype='pdf') as drawing, pymupdf.open() as pdf:
                width, height = output['page_size']
                target = pdf.new_page(width=width, height=height)
                if drawing[0].get_drawings() or drawing[0].get_images():
                    target.show_pdf_page(target.rect, drawing, 0)
                data = pdf.tobytes(deflate=True)
            mime = 'application/pdf'
        elif kind == 'corrected.pdf':
            with pymupdf.open(stream=item['pdf'], filetype='pdf') as source:
                rect = source[page - 1].rect
            with pymupdf.open() as pdf:
                pdf.new_page(width=rect.width, height=rect.height).insert_image(rect, stream=output['png'])
                data = pdf.tobytes(deflate=True)
            mime = 'application/pdf'
        else:
            data = output['svg' if kind == 'trace.svg' else 'png']
            mime = 'image/svg+xml' if kind == 'trace.svg' else 'image/png'
    return Response(data, media_type=mime, headers={
        'Content-Disposition': f'inline; filename="page-{page}-{kind}"',
        'Cache-Control': 'no-store'})
