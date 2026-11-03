"""Seam topology, dense architectural plans, job responsiveness and failures."""
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pymupdf
import pytest
from app.cad import cad_clean, thin, paths_from_skeleton
from app.tiling import Workspace, tiled_thin, owned_paths, tiles, TILE_SIZE
from app.processing import png
from test_app import client, upload


@pytest.mark.parametrize('size', [16, 31, 64])
def test_halo_exchange_matches_whole_page_thinning(size):
    mask = np.zeros((143, 197), np.uint8)
    cv2.rectangle(mask, (4, 6), (190, 133), 1, 13)
    cv2.line(mask, (1, 40), (195, 103), 1, 7)
    cv2.circle(mask, (98, 70), 34, 1, 5)
    with Workspace() as workspace:
        actual, _ = tiled_thin(mask, workspace, size=size)
        assert np.array_equal(actual, thin(mask))
        # Every original skeleton edge is output once, including seam diagonals.
        def edges(paths):
            return [tuple(sorted((a, b))) for path in paths for a, b in zip(path, path[1:])]
        expected = edges(paths_from_skeleton(actual.astype(bool)))
        result = edges([p for box in tiles(mask.shape, size) for p in owned_paths(actual, box, paths_from_skeleton)])
        assert sorted(result) == sorted(expected)
        assert len(result) == len(set(result))


def test_text_and_parallel_lines_across_tile_boundaries():
    gray = np.full((300, 500), 255, np.uint8)
    for x in (63, 64, 127, 128, 383, 384):
        cv2.line(gray, (x, 0), (x, 299), 0, 1)
    for y in (63, 64, 127, 128):
        cv2.line(gray, (0, y), (499, y), 0, 1)
    cv2.putText(gray, 'A1.25', (238, 220), cv2.FONT_HERSHEY_SIMPLEX, .5, 0, 1)
    a = cad_clean(gray, deskew=False, tile_size=64)
    b = cad_clean(gray, deskew=False, tile_size=384)
    decode = lambda output: cv2.imdecode(np.frombuffer(output['png'], np.uint8), 0)
    assert np.array_equal(decode(a) < 100, decode(b) < 100)
    assert np.array_equal(decode(a)[200:225, 235:290] < 100, gray[200:225, 235:290] < 100)
    assert a['added_pixels'] == 0
    assert a['width'] == 500 and a['height'] == 300


def floor_plan(h=1600, w=2400):
    gray = np.full((h, w), 255, np.uint8)
    for y in range(20, h-10, 48):
        cv2.line(gray, (15, y), (w-16, y), 0, 5)
        cv2.line(gray, (15, y+10), (w-16, y+10), 190, 1)
    for x in range(20, w-10, 80):
        cv2.line(gray, (x, 15), (x, h-16), 0, 5)
        cv2.putText(gray, '1200', (x+8, 155), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
    cv2.ellipse(gray, (w//2, h//2), (140, 90), 0, 0, 270, 0, 1)
    return gray


def wait_job(base, job_id, timeout=60):
    deadline = time.monotonic()+timeout
    percentages = []
    while time.monotonic() < deadline:
        response = client.get(f'{base}/jobs/{job_id}')
        assert response.status_code == 200
        data = response.json()
        percentages.append(data['percent'])
        if data['state'] != 'running':
            assert percentages == sorted(percentages)
            return data
        time.sleep(.01)
    pytest.fail('Background processing exceeded test deadline')


def test_large_floor_plan_job_and_vector_exports():
    from app.main import store
    gray = floor_plan()
    assert np.count_nonzero(gray < 220) > 300_000
    token = upload()
    with pymupdf.open() as doc:
        page = doc.new_page(width=1200, height=800)
        page.insert_image(page.rect, stream=png(gray))
        store[token]['pdf'] = doc.tobytes()
    base = f'/api/documents/{token}/pages/1'
    job = client.post(base+'/jobs', json={'mode':'cad_clean', 'deskew':False})
    assert job.status_code == 202
    result = wait_job(base, job.json()['id'])
    assert result['state'] == 'done', result
    assert result['percent'] == 100
    metrics = result['result']
    assert metrics['foreground_pixels'] > 300_000 and metrics['tiles'] > 20
    assert metrics['width'] == 2400 and metrics['height'] == 1600
    root = ET.fromstring(client.get(base+'/trace.svg').content)
    assert root.find('{http://www.w3.org/2000/svg}image') is not None
    assert metrics['mean_width_error_px'] == 0
    assert metrics['fallback_pixel_retention'] == 1
    corrected = cv2.imdecode(np.frombuffer(client.get(base+'/corrected.png').content, np.uint8), 0)
    # Horizontal dimension line crosses every seam, including real intersections.
    assert np.all(corrected[30, 25:2375] < 240)
    with pymupdf.open(stream=client.get(base+'/corrected.pdf').content, filetype='pdf') as doc:
        assert doc[0].rect == pymupdf.Rect(0, 0, 1200, 800)
        assert doc[0].get_images()
    client.delete(f'/api/documents/{token}')


def test_progress_is_available_while_worker_runs_and_no_parallel_jobs(monkeypatch):
    import app.main as main
    entered, release = threading.Event(), threading.Event()
    original = main.compute
    def slow(item, page, options, progress=None):
        progress('境界統合のテスト', 45, 3, 8)
        entered.set()
        assert release.wait(10)
        return original(item, page, options, progress)
    monkeypatch.setattr(main, 'compute', slow)
    token = upload()
    base = f'/api/documents/{token}/pages/1'
    try:
        job = client.post(base+'/jobs', json={'mode':'cad_clean', 'deskew':False}).json()
        assert entered.wait(5)
        progress = client.get(base+'/jobs/'+job['id']).json()
        assert progress['state'] == 'running' and progress['percent'] == 45
        assert progress['completed'] == 3 and progress['total'] == 8
        assert client.post(base+'/jobs', json={}).status_code == 409
        assert client.post(base+'/process', json={}).status_code == 409
        assert client.get(base+'/trace.svg').status_code == 409  # stale output unavailable
    finally:
        release.set()
        if entered.is_set():
            assert wait_job(base, job['id'])['state'] == 'done'
        client.delete(f'/api/documents/{token}')


@pytest.mark.parametrize('exception,fragment', [(MemoryError(), 'メモリ'), (OSError('disk full'), '一時ファイル'), (RuntimeError('private internals'), '処理に失敗')])
def test_failed_jobs_report_japanese_and_release_slot(monkeypatch, exception, fragment):
    import app.main as main
    def fail(*args, **kwargs):
        raise exception
    token = upload()
    base = f'/api/documents/{token}/pages/1'
    with monkeypatch.context() as patch:
        patch.setattr(main, 'compute', fail)
        job = client.post(base+'/jobs', json={}).json()
        result = wait_job(base, job['id'])
        assert result['state'] == 'failed' and fragment in result['error']
        assert 'private internals' not in result['error']
        assert client.get(base+'/trace.svg').status_code == 409
    assert client.post(base+'/process', json={}).status_code == 200
    client.delete(f'/api/documents/{token}')


def test_large_drawing_peak_memory_and_bounded_graph():
    script = Path(__file__).with_name('benchmark_tiling.py')
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=True)
    measurements = json.loads(run.stdout)
    assert measurements['foreground_pixels'] > 300_000
    assert measurements['pixels'] > 10_000_000
    assert measurements['peak_rss_mib'] < 650
    assert measurements['max_graph_pixels'] == 0
    assert measurements['max_analysis_pixels'] <= (TILE_SIZE+166)**2


def test_result_cache_budget_evicts_old_outputs(monkeypatch):
    import app.main as main
    old = {'results': {1: {'png': b'12345678', '_stored_at': 1}}}
    new = {'results': {}}
    monkeypatch.setattr(main, 'store', {'old': old, 'new': new})
    monkeypatch.setattr(main, 'RESULT_CACHE_BYTES', 10)
    with main.lock:
        main.publish(new, 1, {'svg': b'abcdef'})
    assert old['results'] == {}
    assert new['results'][1]['svg'] == b'abcdef'
    assert '_stored_at' not in main.summary(new['results'][1])


def test_deleted_document_is_not_restored_by_worker(monkeypatch):
    import app.main as main
    entered, release = threading.Event(), threading.Event()
    original = main.compute
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(main, 'compute', paused)
    token = upload()
    base = f'/api/documents/{token}/pages/1'
    job = client.post(base+'/jobs', json={'mode':'cad_clean'}).json()
    try:
        assert entered.wait(5)
        assert client.delete(f'/api/documents/{token}').status_code == 200
        assert client.get(base+'/jobs/'+job['id']).status_code == 404
    finally:
        release.set()
    assert main.processing_slot.acquire(timeout=5)
    main.processing_slot.release()
    assert token not in main.store


def test_workspaces_are_deleted_on_success_and_failure(monkeypatch, tmp_path):
    import app.tiling as tiling
    import app.cad as cad
    directory = tiling.tempfile.TemporaryDirectory
    monkeypatch.setattr(tiling.tempfile, 'TemporaryDirectory', lambda **kwargs: directory(dir=tmp_path, **kwargs))
    gray = np.full((64, 128), 255, np.uint8)
    gray[32, 3:125] = 0
    cad.cad_clean(gray, deskew=False)
    assert list(tmp_path.iterdir()) == []
    def fail(*args):
        raise RuntimeError('test failure')
    import app.hybrid as hybrid
    monkeypatch.setattr(hybrid, 'image_tag', fail)
    with pytest.raises(RuntimeError):
        cad.cad_clean(gray, deskew=False)
    assert list(tmp_path.iterdir()) == []
