"""Shape-preserving CAD cleanup; skeleton utilities retained for regression tests only."""
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
    return not np.any((canvas != 0) & ~local.astype(bool))


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
              join_threshold=0.1, snap=False, fidelity='faithful', scale=2.0,
              progress=None, tile_size=None):
    from .tiling import TILE_SIZE, MAX_PIXELS, Workspace, ThinningError
    size = tile_size or TILE_SIZE
    if size < 16 or size > TILE_SIZE:
        raise CadLimitError('小領域のサイズが処理可能な範囲外です。')
    started = time.perf_counter()
    report = progress or (lambda *args: None)
    report('傾き推定', 1, 0, 1)
    if gray.size > MAX_PIXELS:
        raise CadLimitError('処理画像が2000万画素を超えています。ページを分割してください。')
    # Bound Hough workspace; rotation is still applied to the original raster.
    ratio = min(1.0, 1600/max(gray.shape))
    preview = cv2.resize(gray, None, fx=ratio, fy=ratio) if ratio < 1 else gray
    angle = skew_angle(preview) if deskew else 0.0
    h, w = gray.shape
    if abs(angle) > 0.05:
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        c, si = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = int(np.ceil(w*c+h*si)), int(np.ceil(h*c+w*si))
        if nw*nh > MAX_PIXELS:
            raise CadLimitError('回転後の画像が2000万画素を超えます。傾き補正を無効にしてください。')
        matrix[:, 2] += ((nw-w)/2, (nh-h)/2)
        gray = cv2.warpAffine(gray, matrix, (nw, nh), borderValue=255)
    try:
        with Workspace() as workspace:
            return _tiled_clean(gray, angle, noise_strength, line_width, join_threshold,
                                snap, fidelity, scale, started, report, size, workspace)
    except ThinningError as exc:
        raise CadLimitError(str(exc)) from exc


def _tiled_clean(gray, angle, noise_strength, line_width, join_threshold, snap,
                 fidelity, scale, started, report, size, workspace):
    from .hybrid import preserve_clean
    return preserve_clean(gray, angle, noise_strength, scale, started, report, size, workspace)
