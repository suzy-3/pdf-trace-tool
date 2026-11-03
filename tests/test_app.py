import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app, store
from app.processing import clean, png, skew_angle, to_svg

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_store():
    store.clear()
    yield
    store.clear()


def scanned_pdf():
    image = np.full((400, 500), 255, np.uint8)
    cv2.rectangle(image, (70, 70), (430, 330), 0, 3)
    cv2.line(image, (70, 200), (430, 200), 0, 3)
    with pymupdf.open() as pdf:
        for _ in range(2):
            page = pdf.new_page(width=250, height=200)
            page.insert_image(page.rect, stream=png(image))
        return pdf.tobytes()


def upload():
    response = client.post('/api/documents', files={'file': ('scan.pdf', scanned_pdf(), 'application/pdf')})
    assert response.status_code == 200
    assert response.json()['pages'] == 2
    return response.json()['id']


def test_full_scan_workflow():
    token = upload()
    base = f'/api/documents/{token}/pages/2'
    original = client.get(base + '/original.png')
    assert original.status_code == 200
    assert cv2.imdecode(np.frombuffer(original.content, np.uint8), 0).shape == (400, 500)
    assert client.get(base + '/trace.svg').status_code == 409
    processed = client.post(base + '/process', json={})
    assert processed.status_code == 200
    corrected = client.get(base + '/corrected.png')
    assert corrected.headers['content-type'] == 'image/png'
    svg = client.get(base + '/trace.svg')
    root = ET.fromstring(svg.content)
    assert root.find('{http://www.w3.org/2000/svg}path').attrib['d']
    result = client.get(base + '/corrected.pdf')
    assert result.headers['content-type'] == 'application/pdf'
    with pymupdf.open(stream=result.content, filetype='pdf') as pdf:
        assert len(pdf) == 1
        assert pdf[0].rect == pymupdf.Rect(0, 0, 250, 200)
        assert len(pdf[0].get_images()) == 1
    assert client.delete(f'/api/documents/{token}').status_code == 200
    assert client.get(base + '/original.png').status_code == 404


@pytest.mark.parametrize('page', [0, 3, -1])
def test_invalid_pages(page):
    token = upload()
    assert client.get(f'/api/documents/{token}/pages/{page}/original.png').status_code == 404
    assert client.post(f'/api/documents/{token}/pages/{page}/process', json={}).status_code == 404


def test_bad_pdf_and_options():
    assert client.post('/api/documents', files={'file': ('bad.pdf', b'broken')}).status_code == 400
    token = upload()
    assert client.post(f'/api/documents/{token}/pages/1/process', json={'min_area': 0}).status_code == 422
    assert client.get('/api/documents/missing/pages/1/original.png').status_code == 404


def test_deskew_and_noise_removal():
    image = np.full((500, 500), 255, np.uint8)
    for y in [100, 200, 300, 400]:
        cv2.line(image, (50, y), (450, y), 0, 3)
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((250, 250), 7, 1), (500, 500), borderValue=255)
    corrected, angle = clean(rotated)
    assert abs(abs(angle) - 7) < 0.5
    assert abs(skew_angle(corrected)) < 0.5
    image[20, 20] = 0
    cleaned, _ = clean(image, deskew=False, sharpen=False)
    assert cleaned[20, 20] == 255
    assert cleaned[100, 250] == 0


def test_blank_and_svg_holes():
    blank = np.full((100, 100), 255, np.uint8)
    cleaned, angle = clean(blank)
    assert angle == 0 and np.all(cleaned == 255)
    assert ET.fromstring(to_svg(cleaned)) is not None
    cv2.rectangle(blank, (10, 10), (90, 90), 0, 3)
    svg = to_svg(blank)
    assert b'fill-rule="evenodd"' in svg
    assert svg.count(b' Z') == 2


def test_limits_and_expiry(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, 'MAX_BYTES', 10)
    assert client.post('/api/documents', files={'file': ('big.pdf', b'x' * 11)}).status_code == 413
    monkeypatch.setattr(main, 'MAX_BYTES', 20 * 1024 * 1024)
    monkeypatch.setattr(main, 'MAX_DOCUMENTS', 1)
    token = upload()
    assert client.post('/api/documents', files={'file': ('scan.pdf', scanned_pdf())}).status_code == 503
    store[token]['created'] -= 3601
    assert client.get(f'/api/documents/{token}/pages/1/original.png').status_code == 404
    upload()


def test_japanese_ui():
    response = client.get('/')
    assert response.status_code == 200
    assert 'lang="ja"' in response.text
    assert '補正・SVG変換' in response.text
    for name in ['app.js', 'style.css']:
        assert client.get('/static/' + name).status_code == 200
