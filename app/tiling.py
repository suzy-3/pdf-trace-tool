"""Bounded tile workspaces and synchronous halo exchange for exact thinning."""
from pathlib import Path
import tempfile
import numpy as np

TILE_SIZE = 384
MAX_PIXELS = 20_000_000  # includes expanded rotation, not a foreground/graph limit
MAX_SVG_BYTES = 64 * 1024 * 1024


class ThinningError(ValueError):
    """Thinning exceeded the finite iteration budget."""


def tiles(shape, size=TILE_SIZE):
    h, w = shape
    for y in range(0, h, size):
        for x in range(0, w, size):
            yield y, x, min(y+size, h), min(x+size, w)


def window(shape, box, halo):
    y, x, b, r = box
    h, w = shape
    top, left, bottom, right = max(0, y-halo), max(0, x-halo), min(h, b+halo), min(w, r+halo)
    return (slice(top, bottom), slice(left, right)), (slice(y-top, b-top), slice(x-left, r-left))


class Workspace:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pdf-trace-')
        self.maps = []
        return self

    def array(self, name, shape, dtype=np.uint8):
        data = np.memmap(Path(self.temp.name)/name, mode='w+', shape=shape, dtype=dtype)
        data[:] = 0
        self.maps.append(data)
        return data

    def __exit__(self, *args):
        for data in self.maps:
            data._mmap.close()
        self.temp.cleanup()


def deletion_mask(a, phase):
    p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
         a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
    neighbors = sum(p)
    transitions = sum((p[i] == 0) & (p[(i+1) % 8] == 1) for i in range(8))
    if phase == 0:
        gate = (p[0]*p[2]*p[4] == 0) & (p[2]*p[4]*p[6] == 0)
    else:
        gate = (p[0]*p[2]*p[6] == 0) & (p[0]*p[4]*p[6] == 0)
    return (a[1:-1, 1:-1] == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & gate


def tiled_thin(mask, workspace, report=lambda *args: None, size=TILE_SIZE):
    """Every subiteration reads one immutable full state, writes disjoint cores.

    Unlike independently thinning overlapping tiles, this is equivalent to whole
    image Zhang-Suen, even when a thick wall crosses many tile boundaries.
    Only the fixed-size halo workspace and deletion mask live in RAM.
    """
    h, w = mask.shape
    current = workspace.array('thin-a', (h+2, w+2))
    other = workspace.array('thin-b', (h+2, w+2))
    boxes = list(tiles(mask.shape, size))
    for y, x, b, r in boxes:
        current[y+1:b+1, x+1:r+1] = mask[y:b, x:r]
    for iteration in range(256):
        changed = False
        for phase in (0, 1):
            for index, (y, x, b, r) in enumerate(boxes):
                # +1 border in current makes a 1-pixel halo available at page edges.
                local = np.array(current[y:b+2, x:r+2])
                remove = deletion_mask(local, phase)
                core = local[1:-1, 1:-1].copy()
                core[remove] = 0
                other[y+1:b+1, x+1:r+1] = core
                changed |= bool(remove.any())
                report('細線化', 20 + 40*(1-1/(iteration+2)), index+1, len(boxes))
            current, other = other, current
        if not changed:
            return current[1:-1, 1:-1], iteration+1
    raise ThinningError('太い塗り潰し領域が多く細線化が収束しません。線図のページを選択してください。')


def owned_paths(skeleton, box, trace):
    """A shared edge belongs to the tile containing its upper/left endpoint.

    Both sides retain the exact global endpoint. No stitching by proximity, no
    extrapolation, and no duplicated seam strokes are necessary.
    """
    region, _ = window(skeleton.shape, box, 2)
    sy, sx = region[0].start, region[1].start
    y, x, b, r = box
    for local in trace(skeleton[region].astype(bool)):
        path = [(px+sx, py+sy) for px, py in local]
        run = []
        for a, z in zip(path, path[1:]):
            owner = min((a[1], a[0]), (z[1], z[0]))
            if y <= owner[0] < b and x <= owner[1] < r:
                if not run:
                    run.append(a)
                run.append(z)
            elif run:
                yield run
                run = []
        if run:
            yield run
