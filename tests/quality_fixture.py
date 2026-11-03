"""Public synthetic degraded plan. Contains no customer drawing data."""
import cv2
import numpy as np
import pymupdf


def old_plan():
    im = np.full((900, 1200), 255, np.uint8)
    cv2.rectangle(im, (65, 65), (1130, 825), 15, 17)
    cv2.line(im, (65, 405), (1130, 405), 15, 13)
    cv2.line(im, (630, 65), (630, 825), 15, 15)
    # Door arcs attach to walls, windows, endpoints, close parallel dimension lines.
    cv2.ellipse(im, (630, 405), (110, 110), 0, 0, 90, 30, 2)
    cv2.line(im, (630, 405), (740, 405), 20, 2)
    for y in (35, 43):
        cv2.line(im, (70, y), (1120, y), 60, 1)
    for x in (270, 280, 290):
        cv2.line(im, (x, 65), (x, 125), 45, 2)
    cv2.line(im, (100, 740), (500, 675), 50, 2)
    # Render Japanese glyphs using MuPDF's bundled CJK font; no external font download.
    with pymupdf.open() as doc:
        page = doc.new_page(width=1200, height=900)
        for text, point in [('寝室', (160, 250)), ('台所', (780, 250)), ('食堂', (160, 590))]:
            page.insert_text(point, text, fontsize=38, fontname='japan', color=(0, 0, 0))
        pix = page.get_pixmap(colorspace=pymupdf.csGRAY)
        glyphs = np.frombuffer(pix.samples, np.uint8).reshape(900, 1200)
        im = np.minimum(im, glyphs)
    gradient = np.linspace(0, 17, im.shape[1])[None, :]
    im = np.clip(cv2.GaussianBlur(im, (3, 3), .55).astype(float)-gradient, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(5)
    for x, y in rng.integers((90, 100), (1100, 800), size=(110, 2)):
        cv2.circle(im, (int(x), int(y)), 1, 175, -1)
    # Scan compression and slight skew already present in the input.
    _, encoded = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 72])
    im = cv2.imdecode(encoded, 0)
    matrix = cv2.getRotationMatrix2D((600,450), .32, 1)
    im = cv2.warpAffine(im, matrix, (1200,900), borderValue=255)
    regions = {'wall': [50, 50, 1150, 85], 'junction': [600,375,665,440],
               'bedroom_text': [145,200,260,265], 'kitchen_text': [760,200,900,265],
               'dining_text': [145,540,260,605], 'door_arc': [635,420,750,525]}
    return im, regions
