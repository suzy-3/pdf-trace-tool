"""Synthetic geometry checks independent of centerline implementation details."""
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pymupdf
import pytest
from app.cad import cad_clean, thin, paths_from_skeleton
from app.processing import clean, png
from test_app import client, upload


def plan():
    image = np.full((500, 640), 255, np.uint8)
    cv2.line(image, (30, 40), (600, 40), 0, 1)  # thin dimension line
    cv2.line(image, (30, 75), (600, 75), 0, 7)  # thick wall
    for y in (110, 117):
        cv2.line(image, (30, y), (600, y), 0, 1)  # close parallel lines
    for start, end in ((30, 200), (205, 400), (405, 600)):
        cv2.line(image, (start, 150), (end, 150), 0, 1)  # genuine gaps
    cv2.line(image, (30, 190), (220, 270), 0, 1)
    cv2.ellipse(image, (360, 250), (75, 60), 0, 0, 270, 0, 1)
    cv2.putText(image, 'A1.25', (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
    cv2.line(image, (30, 370), (600, 370), 210, 1)  # faint line
    cv2.line(image, (500, 20), (500, 420), 0, 1)  # intersections
    image[450, 30] = 0  # ambiguous dot: retain by default
    gradient = np.linspace(0, 22, image.shape[1])[None, :]
    return np.clip(image.astype(float)-gradient, 0, 255).astype(np.uint8)


def decoded(output):
    return cv2.imdecode(np.frombuffer(output['png'], np.uint8), 0)


def svg_points(output):
    root = ET.fromstring(output['svg'])
    path = root.find('{http://www.w3.org/2000/svg}path')
    assert path.attrib['fill'] == 'none'
    assert path.attrib['stroke'] == 'black'
    return path.attrib


def test_geometry_and_no_invented_centerlines():
    original = plan()
    result = cad_clean(original, deskew=False)
    image = decoded(result)
    attrs = svg_points(result)
    assert float(attrs['stroke-width']) == 1.0
    # Test strokes rather than relying solely on raster similarity.
    assert attrs['d'].count('M') > 10
    for y in (40, 75, 110, 117, 150, 370):
        assert (image[y-1:y+2, 40:190] < 200).any(axis=0).mean() > .98
    assert (image[72:79, 40:190] < 200).sum(axis=0).max() <= 2
    for x in (202, 402):
        assert np.all(image[149:152, x] == 255)  # no inferred gap connection
    assert image[40, 500] < 200  # intersection unchanged
    assert (image[316:334, 30:80] < 200).sum() > 40  # glyph strokes retained
    assert image[450, 30] < 240  # no removal of punctuation by default
    assert result['added_pixels'] == 0
    # Vector centers must lie on measured source ink, including every segment.
    import re
    centers = np.zeros(original.shape, np.uint8)
    for subpath in attrs['d'].split('M')[1:]:
        coords = np.array([[float(n) - .5 for n in pair.split(',')]
                           for pair in re.findall(r'[0-9.]+,[0-9.]+', subpath)], np.int32)
        cv2.polylines(centers, [coords], False, 1, 1)
    assert not np.any((centers > 0) & (original >= 220))
    # Main wall center differs by at most one pixel; thin endpoints retained.
    ys, xs = np.nonzero(centers[70:81, 40:190])
    assert np.max(np.abs(ys + 70 - 75)) <= 1
    assert centers[40, 30] and centers[40, 600]
    assert centers[110, 300] and centers[117, 300]


@pytest.mark.parametrize('fidelity,snap', [('faithful', False), ('clean', False), ('clean', True)])
def test_diagonal_curve_and_gaps_preserved(fidelity, snap):
    image = plan()
    result = cad_clean(image, deskew=False, fidelity=fidelity, snap=snap, join_threshold=1)
    out = decoded(result)
    # All output center paths have nearby source support, including straightening mode.
    assert result['added_pixels'] == 0
    for x, y in ((125, 230), (435, 250), (360, 310)):
        assert (out[y-2:y+3, x-2:x+3] < 200).any()
    assert np.all(out[150, 202:204] == 255)


def test_rotation_expands_canvas_and_keeps_corners():
    image = np.full((400, 500), 255, np.uint8)
    for y in (60, 130, 200, 270, 340):
        cv2.line(image, (30, y), (470, y+30), 0, 1)
    image[0:3, 0:3] = 0
    image[-3:, -3:] = 0
    result = cad_clean(image)
    assert abs(result['angle']) > 2
    assert result['width'] > 500 and result['height'] > 400
    reference = cv2.imdecode(np.frombuffer(result['reference'], np.uint8), 0)
    assert (reference < 200).sum() >= (image < 200).sum() * .9
    assert decoded(result).shape == reference.shape


def test_blank_loops_dots_and_noise_opt_in():
    blank = np.full((100, 100), 255, np.uint8)
    assert np.all(decoded(cad_clean(blank, deskew=False)) == 255)
    blank[20, 20] = 0
    assert decoded(cad_clean(blank, deskew=False))[20, 20] < 240
    assert decoded(cad_clean(blank, deskew=False, noise_strength=1))[20, 20] == 255
    cv2.circle(blank, (50, 50), 20, 0, 1)
    out = cad_clean(blank, deskew=False)
    assert out['paths'] >= 1 and out['protected_components'] >= 1
    assert len(paths_from_skeleton(thin(blank < 100))) >= 2


def test_cad_api_vector_pdf_and_reviews():
    token = upload()
    base = f'/api/documents/{token}/pages/1'
    response = client.post(base+'/process', json={'mode': 'cad_clean', 'deskew': False})
    assert response.status_code == 200, response.text
    assert response.json()['mode'] == 'cad_clean'
    for kind in ('corrected.png', 'reference.png', 'difference.png', 'trace.svg'):
        assert client.get(base+'/'+kind).status_code == 200
    with pymupdf.open(stream=client.get(base+'/corrected.pdf').content, filetype='pdf') as doc:
        assert doc[0].rect == pymupdf.Rect(0, 0, 250, 200)
        assert doc[0].get_images() == []
        assert doc[0].get_drawings()  # actual vector strokes
    assert client.post(base+'/process', json={'mode': 'unknown'}).status_code == 422
    assert client.post(base+'/process', json={'line_width': -1}).status_code == 422
    assert client.post(base+'/process', json={'join_threshold': 2}).status_code == 422
    client.delete(f'/api/documents/{token}')


def test_cad_output_scale_in_pdf_points():
    image = np.full((120, 220), 255, np.uint8)
    image[60, 20:200] = 0
    for scale in (1, 2):
        raster = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        result = cad_clean(raster, deskew=False, scale=scale)
        assert float(svg_points(result)['stroke-width']) / scale == .5
        assert result['width'] / scale == 220


def test_cad_thinning_convergence_budget():
    from app.main import store
    token = upload()
    image = np.zeros((1000, 1000), np.uint8)
    with pymupdf.open() as doc:
        doc.new_page(width=500, height=500).insert_image(pymupdf.Rect(0, 0, 500, 500), stream=png(image))
        store[token]['pdf'] = doc.tobytes()
    response = client.post(f'/api/documents/{token}/pages/1/process', json={'mode':'cad_clean', 'deskew':False})
    assert response.status_code == 413
    assert '収束' in response.json()['detail']
    client.delete(f'/api/documents/{token}')


def test_small_glyphs_remain_original_pixels_in_svg_pdf():
    image = np.full((120, 220), 255, np.uint8)
    cv2.putText(image, 'A1.25', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, .45, 0, 1)
    output = cad_clean(image, deskew=False)
    rendered = decoded(output)
    assert np.array_equal(rendered < 100, image < 100)
    assert b'data:image/png;base64,' in output['svg']
    from app.main import store
    token = upload()
    with pymupdf.open() as doc:
        page = doc.new_page(width=110, height=60)
        page.insert_image(page.rect, stream=png(image))
        store[token]['pdf'] = doc.tobytes()
    base = f'/api/documents/{token}/pages/1'
    assert client.post(base+'/process', json={'mode':'cad_clean', 'deskew':False}).status_code == 200
    with pymupdf.open(stream=client.get(base+'/corrected.pdf').content, filetype='pdf') as doc:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY)
        actual = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        assert np.array_equal(actual < 100, image < 100)
    client.delete(f'/api/documents/{token}')


def test_rotated_pdf_preserves_scale_and_expanded_aspect_ratio():
    from app.main import store
    token = upload()
    image = np.full((300, 400), 255, np.uint8)
    for y in (40, 100, 160, 220):
        cv2.line(image, (20, y), (380, y+30), 0, 1)
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=150)
        page.insert_image(page.rect, stream=png(image))
        store[token]['pdf'] = doc.tobytes()
    base = f'/api/documents/{token}/pages/1'
    response = client.post(base+'/process', json={'mode':'cad_clean'})
    assert response.status_code == 200
    data = response.json()
    assert abs(data['angle']) > 2
    with pymupdf.open(stream=client.get(base+'/corrected.pdf').content, filetype='pdf') as doc:
        assert doc[0].rect.width == pytest.approx(data['width']/2)
        assert doc[0].rect.height == pytest.approx(data['height']/2)
        assert doc[0].rect.width > 200 and doc[0].rect.height > 150
    client.delete(f'/api/documents/{token}')
