"""Conservative raster-supported centerlines; no gap completion or OCR inference."""
import base64
import time

import cv2
import numpy as np
from .processing import png, skew_angle


class CadLimitError(ValueError):
    """Input exceeds the conservative processing budget."""


def thin(mask):
    """Zhang-Suen thinning, preserving connectivity and endpoints."""
    a = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            neighbors = sum(p)
            transitions = sum((p[i] == 0) & (p[(i + 1) % 8] == 1) for i in range(8))
            if phase == 0:
                gate = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                gate = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = (a[1:-1, 1:-1] == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & gate
            if remove.any():
                a[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return a[1:-1, 1:-1].astype(bool)


def paths_from_skeleton(skeleton):
    """Visit each undirected graph edge once, including closed loops and dots."""
    pixels = set(zip(*np.nonzero(skeleton)))
    def neighbors(p):
        y, x = p
        result = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = y + dy, x + dx
            # Avoid diagonal shortcuts around an existing orthogonal connection.
            if q in pixels and not (dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels)):
                result.append(q)
        return result
    graph = {p: neighbors(p) for p in pixels}
    visited = set()
    paths = []
    for start in sorted(pixels, key=lambda p: (len(graph[p]) == 2, p)):
        if not graph[start]:
            paths.append([(start[1], start[0]), (start[1], start[0])])
        for nxt in graph[start]:
            edge = frozenset((start, nxt))
            if edge in visited:
                continue
            visited.add(edge)
            points = [start, nxt]
            prev, cur = start, nxt
            while len(graph[cur]) == 2 and cur != start:
                other = next(p for p in graph[cur] if p != prev)
                edge = frozenset((cur, other))
                if edge in visited:
                    break
                visited.add(edge)
                points.append(other)
                prev, cur = cur, other
            paths.append([(x, y) for y, x in points])
    return paths


def supported(points, ink):
    """Require every proposed centerline pixel to have source ink support."""
    points = np.asarray(points, np.int32)
    x, y = points.min(axis=0)
    right, bottom = points.max(axis=0)
    local = ink[y:bottom+1, x:right+1]
    canvas = np.zeros(local.shape, np.uint8)
    cv2.polylines(canvas, [points - (x, y)], False, 1, 1)
    return not np.any((canvas != 0) & ~local)


def vector_svg(paths, width, height, stroke, protected=None):
    data = ' '.join('M' + ' L'.join(f'{x + 0.5:g},{y + 0.5:g}' for x, y in points) for points in paths)
    image = ''
    if protected is not None and np.any(protected[:, :, 3]):
        encoded = base64.b64encode(png(protected)).decode('ascii')
        image = f'<image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><path fill="none" stroke="black" '
            f'stroke-width="{stroke:.3f}" stroke-linecap="round" stroke-linejoin="round" d="{data}"/>{image}</svg>').encode()


def cad_clean(gray, deskew=True, noise_strength=0, line_width=0.5,
              join_threshold=0.1, snap=False, fidelity='faithful', scale=2.0):
    started = time.perf_counter()
    # Parameters in PDF points: renderer is normally 2 px/pt, supplied by caller.
    return _clean(gray, deskew, noise_strength, line_width, join_threshold, snap,
                  fidelity, scale, started)


def _clean(gray, deskew, noise_strength, line_width, join_threshold, snap,
           fidelity, scale, started):
    angle = skew_angle(gray) if deskew else 0.0
    h, w = gray.shape
    if abs(angle) > 0.05:
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = int(np.ceil(w*c + h*s)), int(np.ceil(h*c + w*s))
        matrix[:, 2] += ((nw-w)/2, (nh-h)/2)
        gray = cv2.warpAffine(gray, matrix, (nw, nh), borderValue=255)
    h, w = gray.shape
    if h*w > 3_000_000:
        raise CadLimitError('回転後のCAD清書は300万画素以下です。ページを分割してください。')
    # Closing estimates paper illumination without blurring foreground strokes.
    kernel = max(3, int(round(12 * scale)) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    contrast = paper.astype(np.int16) - gray.astype(np.int16)
    ink = contrast >= (6 if fidelity == 'faithful' else 10)
    ink |= gray < 100  # keep large solid black regions
    before = ink.copy()
    if noise_strength:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
        # Never remove elongated small components; default removes nothing.
        small = (stats[:, cv2.CC_STAT_AREA] <= noise_strength * scale**2)
        small &= (stats[:, cv2.CC_STAT_WIDTH] <= 2*scale) & (stats[:, cv2.CC_STAT_HEIGHT] <= 2*scale)
        small[0] = False
        ink[small[labels]] = False
    if int(ink.sum()) > 300_000:
        raise CadLimitError('CAD清書は前景30万画素以下です。ページを分割してください。')
    # Protect small components as original raster pixels: punctuation/glyphs must
    # not be reshaped by thinning. This is geometric protection, not text OCR.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    small = (stats[:, cv2.CC_STAT_HEIGHT] <= 12*scale)
    small &= (stats[:, cv2.CC_STAT_WIDTH] <= 40*scale)
    small[0] = False
    protected_mask = small[labels] & ink
    protected = np.zeros((h, w, 4), np.uint8)
    protected[:, :, :3] = gray[:, :, None]
    protected[:, :, 3] = protected_mask.astype(np.uint8)*255
    skeleton = thin(ink & ~protected_mask)
    paths = paths_from_skeleton(skeleton)
    result = []
    for path in paths:
        points = np.asarray(path, np.int32)
        # Short glyphs and curves remain unsmoothed in faithful mode.
        if fidelity == 'clean' and len(points) >= 20*scale:
            candidate = cv2.approxPolyDP(points.reshape(-1, 1, 2), min(join_threshold*scale, 0.75*scale), False).reshape(-1, 2)
            if supported(candidate, ink):
                points = candidate
            if snap and len(points) == 2:
                proposed = points.copy()
                delta = np.abs(points[1]-points[0])
                axis = int(np.argmin(delta))
                if delta[axis] <= min(join_threshold*scale, 0.75*scale):
                    proposed[:, axis] = round(float(points[:, axis].mean()))
                    if supported(proposed, ink):
                        points = proposed
        # Exact collinear reduction changes no geometry, including short glyphs.
        if len(points) > 2:
            delta = np.diff(points, axis=0)
            corners = np.any(delta[1:] != delta[:-1], axis=1)
            points = points[np.r_[True, corners, True]]
        result.append(points.tolist())
    stroke = line_width * scale
    svg = vector_svg(result, w, h, stroke, protected)
    # Render the very same vector output used by PDF/SVG downloads.
    import pymupdf
    with pymupdf.open(stream=svg, filetype='svg') as doc:
        pix = doc[0].get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
        rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    out_ink = rendered < 240
    tolerance = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    removed = before & ~cv2.dilate(out_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    added = out_ink & ~tolerance
    diff = np.full((h, w, 3), 255, np.uint8)
    diff[before] = (150, 150, 150)
    diff[removed] = (0, 0, 220)  # red: source ink distant from output
    diff[added] = (220, 0, 0)    # blue: output distant from source
    return dict(png=png(rendered), svg=svg, reference=png(gray), difference=png(diff),
                angle=angle, width=w, height=h, paths=len(result), protected_components=int(small.sum()),
                missing_pixels=int(removed.sum()), added_pixels=int(added.sum()),
                seconds=round(time.perf_counter()-started, 3),
                warning='赤は原画から離れた欠落候補、青は追加候補です。文字・太線も必ず原画と比較してください。')
