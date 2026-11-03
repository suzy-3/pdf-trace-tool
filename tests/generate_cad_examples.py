"""Reproduce synthetic review images and process peak-RSS measurements."""
import json
from pathlib import Path
import resource
import sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_cad import plan
from app.cad import cad_clean
from app.processing import clean, png


def main():
    target = ROOT / 'docs' / 'examples'
    target.mkdir(parents=True, exist_ok=True)
    image = plan()
    legacy, _ = clean(image, deskew=False)
    result = cad_clean(image, deskew=False)
    (target/'original.png').write_bytes(png(image))
    (target/'legacy.png').write_bytes(png(legacy))
    (target/'cad-clean.png').write_bytes(result['png'])
    (target/'cad-clean.svg').write_bytes(result['svg'])
    (target/'difference.png').write_bytes(result['difference'])
    panels = []
    for name, data in [('Original', image), ('Legacy', legacy), ('CAD centerlines', cv2.imdecode(np.frombuffer(result['png'], np.uint8), 0))]:
        panel = cv2.copyMakeBorder(data, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(panel, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, 0, 1)
        panels.append(panel)
    (target/'comparison.png').write_bytes(png(np.hstack(panels)))
    rows = []
    for label, raster in [('640x500', image), ('2000x1500', cv2.resize(image, (2000,1500), interpolation=cv2.INTER_NEAREST))]:
        result = cad_clean(raster, deskew=False, scale=raster.shape[1]/320)
        rows.append(dict(case=label, seconds=result['seconds'], paths=result['paths'],
                         missing_pixels=result['missing_pixels'], added_pixels=result['added_pixels'],
                         process_peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1)))
    (target/'measurements.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
