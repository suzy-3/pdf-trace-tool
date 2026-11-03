"""Geometry-gated hybrid cleanup. Never skeletonize walls, glyphs or curves.

Classification is geometric, not architectural/OCR recognition. Unsupported and
ambiguous shapes retain their original grayscale pixels, including broad walls.
"""
import base64
import tempfile
import time

import cv2
import numpy as np
import pymupdf
from .processing import png
from .tiling import tiles, window, MAX_SVG_BYTES


def render_svg(svg):
    with pymupdf.open(stream=svg, filetype='svg') as doc:
        pix = doc[0].get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def image_tag(gray, alpha=None, x=0, y=0):
    data = gray
    if alpha is not None:
        data = np.zeros((*gray.shape, 4), np.uint8)
        data[:, :, :3] = gray[:, :, None]
        data[:, :, 3] = alpha
    encoded = base64.b64encode(png(data)).decode('ascii')
    h, w = gray.shape
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{encoded}"/>'.encode()


def fit_stroke(mask, values, scale):
    """Fit a long strip, validate both boundaries, width, location and endpoints.

    Accept only exact foreground support, not a Hough edge invented across gaps.
    The check uses the actual SVG renderer used by downloads, including caps.
    """
    yy, xx = np.nonzero(mask)
    if len(xx) < 60*scale or max(np.ptp(xx), np.ptp(yy)) < 60*scale:
        return None
    if int(values.max())-int(values.min()) > 3 or int(values.max()) > 180:
        return None  # degraded or pale ink: keep the original, do not threshold it away
    points = np.column_stack((xx, yy)).astype(np.float32)
    vx, vy, cx, cy = map(float, cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel())
    delta = points - (cx, cy)
    along = delta @ (vx, vy)
    across = delta @ (-vy, vx)
    width = float(np.ptp(across)+abs(vx)+abs(vy))
    length = float(np.ptp(along)+1)
    if width > 16*scale or length < max(60*scale, 12*width):
        return None
    # Uniform strip boundaries, no local bulges, wall junctions or door arcs.
    centers, widths = [], []
    for edges in np.linspace(float(along.min()), float(along.max()), 9)[:-1]:
        band = across[(along >= edges) & (along <= edges+length/8)]
        if not len(band):
            return None
        centers.append(float((band.min()+band.max())/2))
        widths.append(float(np.ptp(band)+abs(vx)+abs(vy)))
    if np.ptp(centers) > .25 or np.ptp(widths) > .25:
        return None
    start = np.array((cx, cy)) + np.array((vx, vy))*(float(along.min())-.5)
    end = np.array((cx, cy)) + np.array((vx, vy))*(float(along.max())+.5)
    color = int(np.median(values))
    path = (f'<path fill="none" stroke="rgb({color},{color},{color})" stroke-width="{width:.6f}" '
            f'stroke-linecap="butt" d="M{start[0]+.5:.6f},{start[1]+.5:.6f} '
            f'L{end[0]+.5:.6f},{end[1]+.5:.6f}"/>')
    h, w = mask.shape
    proposal = render_svg((f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'+path+'</svg>').encode())
    if not np.array_equal(proposal < 220, mask):
        return None
    out_y, out_x = np.nonzero(proposal < 220)
    position_error = float(np.hypot(out_x.mean()-xx.mean(), out_y.mean()-yy.mean()))
    if position_error > .25 or np.max(np.abs(proposal[mask].astype(int)-values.astype(int))) > 1:
        return None
    return path, width, position_error


def quality_metrics(reference, result):
    """Raw mask overlap, local connectivity and stroke thickness; no semantic claim."""
    original, output = reference < 200, result < 200
    count = max(1, int(original.sum()))
    missing = int((original & ~output).sum())
    added = int((~original & output).sum())
    before_components = cv2.connectedComponents(original.astype(np.uint8))[0]-1
    after_components = cv2.connectedComponents(output.astype(np.uint8))[0]-1
    # Distance-transform error evaluated on retained source ink, bounded per tile.
    a = cv2.distanceTransform(np.pad(original.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    b = cv2.distanceTransform(np.pad(output.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    width_error = float(np.abs(a[original]-b[original]).sum()*2)
    return dict(source_pixels=int(original.sum()), missing=missing, added=added,
                connectivity_change=abs(before_components-after_components), width_error=width_error,
                passed=missing/count <= .005 and added/count <= .002
                       and before_components == after_components and width_error/count <= .25)


def preserve_clean(gray, angle, noise_strength, scale, started, report, size, workspace):
    from .cad import CadLimitError
    h, w = gray.shape
    boxes = list(tiles(gray.shape, size))
    halo = max(32, int(np.ceil(40*scale))+3)
    fallback = workspace.array('fallback', gray.shape)
    proposed = workspace.array('proposed', gray.shape)
    proposed[:] = gray
    vector_count = wall_pixels = noise_pixels = foreground = max_analysis_pixels = 0
    position_error = 0.0
    widths = []
    with tempfile.TemporaryFile() as fragments:
        for index, box in enumerate(boxes):
            region, core = window(gray.shape, box, halo)
            y, x, b, r = box
            local = gray[region]
            max_analysis_pixels = max(max_analysis_pixels, local.size)
            # Pale/uncertain ink is NOT used for straight-line fitting, but remains in raster.
            mask = local < 220
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            alpha = np.full((b-y, r-x), 255, np.uint8)
            tile = gray[y:b, x:r].copy()
            tile_vectors = []
            for label in range(1, len(stats)):
                left, top, cw, ch, area = stats[label]
                # Fit only an elongated strip. Most walls/junctions/glyphs stay raster.
                if max(cw, ch) < 60*scale or max(cw, ch) < 12*max(1, min(cw, ch)):
                    continue
                component = labels[top:top+ch, left:left+cw] == label
                values = local[top:top+ch, left:left+cw][component]
                fitted = fit_stroke(component, values, scale)
                if fitted is None:
                    continue
                path, width, error = fitted
                # Every tile clips to its disjoint core; shared coordinates don't move.
                ox, oy = region[1].start+int(left), region[0].start+int(top)
                tile_vectors.append(f'<g transform="translate({ox},{oy})">{path}</g>')
                keep = (labels == label)[core]
                alpha[keep] = 0
                vector_count += 1
                widths.append(width/scale)
                position_error = max(position_error, error)
            # Noise handling is separate and opt-in: only tiny isolated gray spots.
            if noise_strength > 0:
                for label in range(1, len(stats)):
                    left, top, cw, ch, area = stats[label]
                    if area > noise_strength*scale**2 or max(cw, ch) > 2*scale:
                        continue
                    values = local[top:top+ch, left:left+cw][labels[top:top+ch, left:left+cw] == label]
                    if values.size and values.min() >= 140:
                        remove = (labels == label)[core]
                        tile[remove] = 255
                        noise_pixels += int(remove.sum())
            proposed[y:b, x:r] = tile
            fallback[y:b, x:r] = ((gray[y:b, x:r] < 220) & (alpha > 0) & (tile == gray[y:b, x:r]))
            distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            wall_pixels += int(((distance >= 3*scale) & mask)[core].sum())
            foreground += int(mask[core].sum())
            clip = f'<defs><clipPath id="core{index}"><rect x="{x}" y="{y}" width="{r-x}" height="{b-y}"/></clipPath></defs>'
            fragments.write(clip.encode())
            fragments.write(f'<g clip-path="url(#core{index})">'.encode())
            fragments.write(''.join(tile_vectors).encode())
            fragments.write(b'</g>')
            fragments.write(image_tag(tile, alpha, x, y))
            if fragments.tell() > MAX_SVG_BYTES:
                raise CadLimitError('出力が64MBを超えます。ページを分割してください。')
            report('壁・直線候補の検証／原画保護', 3+72*(index+1)/len(boxes), index+1, len(boxes))
        fragments.seek(0)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.encode()
               + fragments.read() + b'</svg>')
    report('形状・線幅の品質検証', 77, 0, len(boxes))
    rendered = render_svg(svg)
    totals = dict(source_pixels=0, missing=0, added=0, connectivity_change=0, width_error=0.0)
    passed = True
    fallback_count = unchanged_fallback = 0
    for index, (y, x, b, r) in enumerate(boxes):
        metric = quality_metrics(gray[y:b, x:r], rendered[y:b, x:r])
        for key in totals:
            totals[key] += metric[key]
        passed &= quality_metrics(proposed[y:b, x:r], rendered[y:b, x:r])['passed']
        kept = fallback[y:b, x:r].astype(bool)
        fallback_count += int(kept.sum())
        unchanged_fallback += int((kept & (gray[y:b, x:r] == rendered[y:b, x:r])).sum())
        report('形状・線幅の品質検証', 78+10*(index+1)/len(boxes), index+1, len(boxes))
    attempted = dict(totals)
    if not passed:
        # Reject the entire proposal, including optional noise removal, not just warn.
        with tempfile.TemporaryFile() as fragments:
            for y, x, b, r in boxes:
                fragments.write(image_tag(gray[y:b, x:r], x=x, y=y))
            fragments.seek(0)
            svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.encode()+fragments.read()+b'</svg>'
        rendered = gray.copy()
        fallback[:] = gray < 220
        fallback_count = foreground
        unchanged_fallback = foreground
    diff = workspace.array('difference', (h, w, 3))
    review = workspace.array('review', (h, w, 3))
    for y, x, b, r in boxes:
        original, output = gray[y:b, x:r] < 200, rendered[y:b, x:r] < 200
        canvas = np.full((b-y, r-x, 3), 255, np.uint8)
        canvas[original] = (150, 150, 150)
        canvas[original & ~output] = (0, 0, 220)
        canvas[~original & output] = (220, 0, 0)
        diff[y:b, x:r] = canvas
        canvas = cv2.cvtColor(rendered[y:b, x:r], cv2.COLOR_GRAY2BGR)
        canvas[fallback[y:b, x:r].astype(bool)] = (20, 160, 230)
        review[y:b, x:r] = canvas
    unresolved = fallback_count/max(1, foreground)
    warning = ('要確認：未認識の壁・文字・建具は原画を保持しています。完全なCAD化ではありません。'
               if unresolved > .05 else '形状検証済み。文字・寸法関係を原画と比較してください。')
    if not passed:
        warning = '要確認：品質基準を満たさなかったため、清書候補を破棄して全体を原画に戻しました。'
    report('品質情報と出力の保存', 96, len(boxes), len(boxes))
    output = dict(png=png(rendered), svg=svg, reference=png(gray), difference=png(diff), review=png(review),
                  angle=angle, width=w, height=h, paths=vector_count if passed else 0,
                  protected_components=0, foreground_pixels=foreground, wall_interior_pixels=wall_pixels,
                  tiles=len(boxes), tile_size=size, max_analysis_pixels=max_analysis_pixels, thinning_iterations=0, engine='hybrid_preserve',
                  missing_pixels=totals['missing'] if passed else 0, added_pixels=totals['added'] if passed else 0,
                  quality_passed=bool(passed), review_required=bool(unresolved > .05 or not passed),
                  fallback_ratio=round(unresolved, 6), fallback_pixel_retention=round(unchanged_fallback/fallback_count, 6) if fallback_count else 1.0,
                  missing_ratio=round(totals['missing']/max(1, totals['source_pixels']), 6) if passed else 0,
                  added_ratio=round(totals['added']/max(1, totals['source_pixels']), 6) if passed else 0,
                  local_connectivity_change=totals['connectivity_change'] if passed else 0,
                  mean_width_error_px=round(totals['width_error']/max(1, totals['source_pixels']), 6) if passed else 0,
                  max_line_position_error_px=round(position_error, 6),
                  measured_line_widths_pt=[round(value, 4) for value in widths[:32]],
                  noise_removed_pixels=noise_pixels if passed else 0, attempted_quality=attempted,
                  seconds=round(time.perf_counter()-started, 3), warning=warning)
    report('完了', 100, len(boxes), len(boxes))
    return output
