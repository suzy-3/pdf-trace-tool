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
    import tempfile
    import pymupdf
    from .tiling import tiles, window, tiled_thin, owned_paths, MAX_SVG_BYTES
    h, w = gray.shape
    boxes = list(tiles(gray.shape, size))
    ink = workspace.array('ink', gray.shape)
    before = workspace.array('before', gray.shape)
    protected = workspace.array('protected', gray.shape)
    kernel = max(3, int(round(12*scale)) | 1)
    # Closing has support twice the kernel radius. Protection needs an entire
    # small connected component even when a glyph straddles the core boundary.
    halo = max(kernel+2, int(np.ceil(40*scale))+3)
    protected_count = 0
    foreground = 0
    for index, box in enumerate(boxes):
        region, core = window(gray.shape, box, halo)
        local = gray[region]
        paper = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        mask = (paper.astype(np.int16)-local.astype(np.int16) >= (6 if fidelity == 'faithful' else 10)) | (local < 100)
        y, x, b, r = box
        before[y:b, x:r] = mask[core]
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if noise_strength:
            remove = (stats[:, cv2.CC_STAT_AREA] <= noise_strength*scale**2)
            remove &= (stats[:, cv2.CC_STAT_WIDTH] <= 2*scale) & (stats[:, cv2.CC_STAT_HEIGHT] <= 2*scale)
            remove[0] = False
            mask[remove[labels]] = False
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        small = (stats[:, cv2.CC_STAT_HEIGHT] <= 12*scale) & (stats[:, cv2.CC_STAT_WIDTH] <= 40*scale)
        small[0] = False
        # A truncated component must never be mistaken for a small glyph.
        if region[0].start > 0:
            small[np.unique(labels[0])] = False
        if region[0].stop < h:
            small[np.unique(labels[-1])] = False
        if region[1].start > 0:
            small[np.unique(labels[:, 0])] = False
        if region[1].stop < w:
            small[np.unique(labels[:, -1])] = False
        protected[y:b, x:r] = (small[labels] & mask)[core]
        ink[y:b, x:r] = mask[core]
        foreground += int(mask[core].sum())
        top = stats[:, cv2.CC_STAT_TOP] + region[0].start
        left = stats[:, cv2.CC_STAT_LEFT] + region[1].start
        protected_count += int((small & (top >= y) & (top < b) & (left >= x) & (left < r)).sum())
        report('前処理・文字保護', 2+18*(index+1)/len(boxes), index+1, len(boxes))
    mask = workspace.array('mask', gray.shape)
    for y, x, b, r in boxes:
        mask[y:b, x:r] = ink[y:b, x:r] & (1-protected[y:b, x:r])
    skeleton, iterations = tiled_thin(mask, workspace, report, size)
    path_count = 0
    # Spool vector text to disk: never retain a full-page Python graph/path list.
    with tempfile.TemporaryFile() as paths, tempfile.TemporaryFile() as images:
        for index, box in enumerate(boxes):
            y, x, b, r = box
            for path in owned_paths(skeleton, box, paths_from_skeleton):
                points = np.asarray(path, np.int32)
                if fidelity == 'clean' and len(points) >= 20*scale:
                    candidate = cv2.approxPolyDP(points.reshape(-1, 1, 2), min(join_threshold*scale, .75*scale), False).reshape(-1, 2)
                    if supported(candidate, ink):
                        points = candidate
                    # Snapping may not move seam/junction endpoints.
                    if snap and len(points) == 2 and all(x+1 < px < r-2 and y+1 < py < b-2 for px, py in points):
                        isolated = all(int(skeleton[py-1:py+2, px-1:px+2].sum()) <= 2 for px, py in points)
                        proposed = points.copy()
                        delta = np.abs(points[1]-points[0])
                        axis = int(np.argmin(delta))
                        if isolated and delta[axis] <= min(join_threshold*scale, .75*scale):
                            proposed[:, axis] = round(float(points[:, axis].mean()))
                            if supported(proposed, ink):
                                points = proposed
                if len(points) > 2:
                    delta = np.diff(points, axis=0)
                    points = points[np.r_[True, np.any(delta[1:] != delta[:-1], axis=1), True]]
                paths.write(('M'+' L'.join(f'{px+.5:g},{py+.5:g}' for px, py in points)+' ').encode())
                path_count += 1
            if np.any(protected[y:b, x:r]):
                rgba = np.zeros((b-y, r-x, 4), np.uint8)
                rgba[:, :, :3] = gray[y:b, x:r, None]
                rgba[:, :, 3] = protected[y:b, x:r]*255
                encoded = base64.b64encode(png(rgba)).decode('ascii')
                images.write(f'<image x="{x}" y="{y}" width="{r-x}" height="{b-y}" href="data:image/png;base64,{encoded}"/>'.encode())
            if paths.tell()+images.tell() > MAX_SVG_BYTES:
                raise CadLimitError('線や画像が複雑すぎて出力が64MBを超えます。ノイズ除去を調整してください。')
            report('中心線の統合', 60+25*(index+1)/len(boxes), index+1, len(boxes))
        paths.seek(0)
        images.seek(0)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><path fill="none" stroke="black" stroke-width="{line_width*scale:.3f}" stroke-linecap="round" stroke-linejoin="round" d="'.encode()
               + paths.read() + b'"/>' + images.read() + b'</svg>')
    report('SVG描画', 86, 0, 1)
    with pymupdf.open(stream=svg, filetype='svg') as doc:
        pix = doc[0].get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
        rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    diff = workspace.array('diff', (h, w, 3))
    missing = added = 0
    for index, box in enumerate(boxes):
        region, core = window(gray.shape, box, 1)
        y, x, b, r = box
        out = rendered[region] < 240
        tolerance = cv2.dilate(ink[region], np.ones((3, 3), np.uint8)).astype(bool)
        removed = before[y:b, x:r].astype(bool) & ~cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)[core]
        extra = out[core] & ~tolerance[core]
        canvas = np.full((b-y, r-x, 3), 255, np.uint8)
        canvas[before[y:b, x:r].astype(bool)] = (150, 150, 150)
        canvas[removed] = (0, 0, 220)
        canvas[extra] = (220, 0, 0)
        diff[y:b, x:r] = canvas
        missing += int(removed.sum())
        added += int(extra.sum())
        report('差分確認', 87+9*(index+1)/len(boxes), index+1, len(boxes))
    report('出力を保存', 97, 0, 1)
    output = dict(png=png(rendered), svg=svg, reference=png(gray), difference=png(diff),
                  angle=angle, width=w, height=h, paths=path_count, protected_components=protected_count,
                  missing_pixels=missing, added_pixels=added, foreground_pixels=foreground,
                  tiles=len(boxes), tile_size=size, thinning_iterations=iterations,
                  seconds=round(time.perf_counter()-started, 3),
                  warning='赤は欠落候補、青は追加候補です。文字・寸法線を原画と比較してください。')
    report('完了', 100, len(boxes), len(boxes))
    return output
