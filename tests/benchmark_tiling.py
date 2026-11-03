"""Separate process for meaningful peak RSS; synthetic near-A0 raster."""
import json
from pathlib import Path
import resource
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.cad as cad
from test_tiling import floor_plan


def main():
    raster = floor_plan(2800, 4000)
    max_graph_pixels = 0
    original = cad.paths_from_skeleton
    def traced(mask):
        nonlocal max_graph_pixels
        max_graph_pixels = max(max_graph_pixels, mask.size)
        return original(mask)
    cad.paths_from_skeleton = traced
    output = cad.cad_clean(raster, deskew=False)
    print(json.dumps(dict(pixels=raster.size, foreground_pixels=output['foreground_pixels'],
                          tiles=output['tiles'], seconds=output['seconds'],
                          max_graph_pixels=max_graph_pixels,
                          peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1))))


if __name__ == '__main__':
    main()
