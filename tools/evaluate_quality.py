"""Local-only quality report. Explicit --output is required for private PDFs.

Synthetic: python tools/evaluate_quality.py --before-revision 68e7d64
Private:   python tools/evaluate_quality.py --input /private/plan.pdf --output /private/report
Optional --regions JSON maps labels to [x0,y0,x1,y1] in the rendered image.
No upload, automatic git add, external service or OCR is involved.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
import cv2
import numpy as np
import pymupdf
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from app.cad import cad_clean
from app.hybrid import quality_metrics
from app.processing import png
from quality_fixture import old_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--regions', type=Path)
    parser.add_argument('--before-revision', help='Explicit trusted repository revision to compare')
    args = parser.parse_args()
    if args.input and not args.output:
        parser.error('Private input requires an explicit private --output directory')
    if args.input:
        with pymupdf.open(args.input) as doc:
            if not 1 <= args.page <= len(doc):
                parser.error('Page is out of range')
            page = doc[args.page-1]
            scale = min(2, (12_000_000/max(1, page.rect.width*page.rect.height))**.5)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        regions = {}
    else:
        gray, regions = old_plan()
        scale = 2
    if args.regions:
        regions = json.loads(args.regions.read_text(encoding='utf-8'))
    target = args.output or ROOT/'work'/'quality-evaluation'
    target.mkdir(parents=True, exist_ok=True)
    output = cad_clean(gray, deskew=False, scale=scale)
    actual = cv2.imdecode(np.frombuffer(output['png'], np.uint8), 0)
    rows = {'after': {k:v for k,v in output.items() if not isinstance(v, bytes)}}
    panels = [('Original', gray)]
    if args.before_revision:
        # Only run explicitly selected repository code, never fetched PDF contents.
        source = subprocess.run(['git', 'show', f'{args.before_revision}:app/cad.py'], cwd=ROOT,
                                check=True, text=True, capture_output=True).stdout
        baseline = types.ModuleType('app._quality_baseline')
        baseline.__package__ = 'app'
        exec(compile(source, 'trusted-baseline-cad.py', 'exec'), baseline.__dict__)
        old = baseline.cad_clean(gray, deskew=False, scale=scale)
        old_image = cv2.imdecode(np.frombuffer(old['png'], np.uint8), 0)
        panels.append(('Before (uniform thinning)', old_image))
        rows['before'] = quality_metrics(gray, old_image)
        (target/'before.png').write_bytes(old['png'])
    panels.append(('After (hybrid preserve)', actual))
    rows['regions'] = {}
    for label, (x0,y0,x1,y1) in regions.items():
        if not (0 <= x0 < x1 <= gray.shape[1] and 0 <= y0 < y1 <= gray.shape[0]):
            raise ValueError(f'Invalid region: {label}')
        reference = gray[y0:y1,x0:x1]
        candidate = actual[y0:y1,x0:x1]
        metric = quality_metrics(reference, candidate)
        metric['unchanged_pixel_ratio'] = float(np.mean(reference == candidate))
        rows['regions'][label] = metric
    images = []
    for label, image in panels:
        panel = cv2.copyMakeBorder(image, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(panel, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .65, 0, 1)
        images.append(panel)
    (target/'comparison.png').write_bytes(png(np.hstack(images)))
    (target/'original.png').write_bytes(png(gray))
    (target/'after.png').write_bytes(output['png'])
    (target/'review.png').write_bytes(output['review'])
    (target/'trace.svg').write_bytes(output['svg'])
    (target/'metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(target), 'missing_ratio': output['missing_ratio'],
                      'width_error_px':output['mean_width_error_px'], 'review_required':output['review_required']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
