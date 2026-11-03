import cv2
import numpy as np
import pymupdf
import pytest
from app.cad import cad_clean
from app.hybrid import render_svg
from quality_fixture import old_plan
from test_app import client, upload


def decode(output):
    return cv2.imdecode(np.frombuffer(output['png'], np.uint8), 0)


def test_degraded_plan_preserves_walls_japanese_text_doors_and_junctions(monkeypatch):
    import app.tiling as tiling
    def forbidden(*args, **kwargs):
        pytest.fail('Hybrid mode must never call skeleton thinning')
    monkeypatch.setattr(tiling, 'tiled_thin', forbidden)
    original, regions = old_plan()
    result = cad_clean(original, deskew=False)
    actual = decode(result)
    assert result['engine'] == 'hybrid_preserve'
    assert result['quality_passed'] and result['review_required']
    assert result['missing_ratio'] == result['added_ratio'] == 0
    assert result['mean_width_error_px'] == result['local_connectivity_change'] == 0
    for x0,y0,x1,y1 in regions.values():
        assert np.array_equal(original[y0:y1,x0:x1], actual[y0:y1,x0:x1])
    assert np.array_equal(original, actual)


def test_measured_wall_width_not_arbitrary_thin_stroke():
    image = np.full((300, 720), 255, np.uint8)
    cv2.rectangle(image, (20,60), (690,78), 0, -1)
    output = cad_clean(image, deskew=False, line_width=.1, snap=True, fidelity='clean')
    assert output['paths'] > 0
    assert all(width == pytest.approx(9.5, abs=.1) for width in output['measured_line_widths_pt'])
    assert output['max_line_position_error_px'] <= .25
    assert np.array_equal(image < 200, decode(output) < 200)
    assert np.array_equal(image < 200, render_svg(output['svg']) < 200)


def test_low_confidence_gaps_pale_strokes_and_connected_text_fall_back():
    image = np.full((300,720),255,np.uint8)
    cv2.line(image,(10,80),(710,80),190,9)
    image[75:86,350:358] = 255
    cv2.putText(image,'A1.25',(300,90),cv2.FONT_HERSHEY_SIMPLEX,1,30,2)
    cv2.ellipse(image,(400,140),(180,70),0,0,180,30,2)
    output = cad_clean(image,deskew=False)
    assert output['review_required']
    assert np.array_equal(image,decode(output))


def test_quality_gate_discards_bad_proposal_and_preserves_source(monkeypatch):
    import app.hybrid as hybrid
    original = np.full((160,320),255,np.uint8)
    cv2.rectangle(original,(20,40),(290,60),0,-1)
    real = hybrid.render_svg
    def broken(svg):
        rendered = real(svg)
        # Fail the final full-page output only; candidate checks may still pass.
        if b'clipPath' in svg:
            rendered[35:65,100:140] = 255
        return rendered
    monkeypatch.setattr(hybrid,'render_svg',broken)
    output = cad_clean(original,deskew=False)
    assert not output['quality_passed'] and output['review_required']
    assert '破棄' in output['warning']
    assert np.array_equal(original,decode(output))
    assert np.array_equal(original,real(output['svg']))
    assert output['attempted_quality']['missing'] > 0


def test_original_preserving_pdf_available_before_processing_and_after_failure():
    token=upload()
    base=f'/api/documents/{token}/pages/1'
    result=client.get(base+'/preserved.pdf')
    assert result.status_code==200
    with pymupdf.open(stream=result.content,filetype='pdf') as doc:
        assert doc[0].rect==pymupdf.Rect(0,0,250,200)
        before=doc[0].get_pixmap(colorspace=pymupdf.csGRAY).samples
    client.post(base+'/process',json={'mode':'cad_clean','deskew':False})
    assert client.get(base+'/review.png').status_code==200
    with pymupdf.open(stream=client.get(base+'/preserved.pdf').content,filetype='pdf') as doc:
        assert doc[0].get_pixmap(colorspace=pymupdf.csGRAY).samples==before
    client.delete(f'/api/documents/{token}')


def test_vector_pdf_keeps_the_measured_wall_band():
    from app.main import store
    token = upload()
    image = np.full((300,720),255,np.uint8)
    cv2.rectangle(image,(20,60),(690,78),0,-1)
    with pymupdf.open() as doc:
        page=doc.new_page(width=360,height=150)
        page.insert_image(page.rect,stream=cv2.imencode('.png',image)[1].tobytes())
        store[token]['pdf']=doc.tobytes()
    base=f'/api/documents/{token}/pages/1'
    response=client.post(base+'/process',json={'mode':'cad_clean','deskew':False})
    assert response.status_code==200 and response.json()['paths']>0
    with pymupdf.open(stream=client.get(base+'/corrected.pdf').content,filetype='pdf') as doc:
        assert doc[0].get_drawings()
        pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2),colorspace=pymupdf.csGRAY)
        actual=np.frombuffer(pix.samples,np.uint8).reshape(300,720)
        assert np.array_equal(actual < 200,image < 200)
        assert doc[0].rect==pymupdf.Rect(0,0,360,150)
    client.delete(f'/api/documents/{token}')
