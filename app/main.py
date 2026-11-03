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


@app.post('/api/documents/{token}/pages/{page}/process')
def process(token: str, page: int, options: Options):
    with lock:
        item = document(token)
        source = page_image(item, page)
        if options.mode == 'cad_clean':
            with pymupdf.open(stream=item['pdf'], filetype='pdf') as doc:
                rect = doc[page - 1].rect
            scale = min(2.0, (MAX_PIXELS / max(1, rect.width * rect.height)) ** 0.5)
            # The Python skeleton graph needs substantially more memory than a raster.
            if source.size > 3_000_000:
                raise HTTPException(413, 'CAD清書は300万画素以下です。ページを分割してください。')
            try:
                output = cad_clean(source, deskew=options.deskew, scale=scale,
                                   **{k: getattr(options, k) for k in
                                      ('line_width', 'noise_strength', 'join_threshold', 'snap', 'fidelity')})
            except CadLimitError as exc:
                raise HTTPException(413, str(exc)) from exc
            output['mode'] = 'cad_clean'
            output['page_size'] = (output['width']/scale, output['height']/scale)
            item['results'][page] = output
            return {k: v for k, v in output.items() if k not in
                    ('png', 'svg', 'reference', 'difference', 'page_size')}
        image, angle = clean(source, deskew=options.deskew, denoise=options.denoise,
                             sharpen=options.sharpen, min_area=options.min_area)
        item['results'][page] = {'png': png(image), 'svg': to_svg(image)}
    return {'angle': round(angle, 2), 'width': image.shape[1], 'height': image.shape[0]}


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
