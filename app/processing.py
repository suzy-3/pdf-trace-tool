"""Raster cleanup and filled contour tracing (not CAD centerline recognition)."""
import math

import cv2
import numpy as np


def skew_angle(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, 50,
                            minLineLength=max(30, gray.shape[1] // 8), maxLineGap=15)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            angle = (angle + 45) % 90 - 45
            if abs(angle) <= 15:
                angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def clean(gray, deskew=True, denoise=True, sharpen=True, min_area=6):
    angle = skew_angle(gray) if deskew else 0.0
    height, width = gray.shape
    if abs(angle) > 0.05:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        gray = cv2.warpAffine(gray, matrix, (width, height), borderValue=255)
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    if sharpen:
        gray = cv2.addWeighted(gray, 1.7, cv2.GaussianBlur(gray, (0, 0), 1), -0.7, 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if denoise:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        keep = np.zeros(count, dtype=np.uint8)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area) * 255
        ink = keep[labels]
    return 255 - ink, angle


def to_svg(image):
    height, width = image.shape
    contours, _ = cv2.findContours(255 - image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        points = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        paths.append('M' + ' L'.join(f'{x},{y}' for x, y in points) + ' Z')
    data = ' '.join(paths)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><path fill="black" fill-rule="evenodd" '
            f'd="{data}"/></svg>').encode()


def png(image):
    return cv2.imencode('.png', image)[1].tobytes()
