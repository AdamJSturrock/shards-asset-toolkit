# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Trace leg columns up from foot contacts (landmark helper for the jig).

python3 scripts/critter/trace_legs.py <critter> x,y [x,y ...] [--r 0.07] [--grow 2.2]

For each foot contact (x, y) in rig space, walks up the mesh in 0.02 z
slices: the slice's vertices within `r` of the current axis give the new
axis point (so a tilted leg is followed), and the climb stops where the
slice's vertices within 2r spread wider than `grow` x the leg's median
radius, which is where the leg enters the body. Prints the axis at 0, 25,
50, 75 and 100% of the leg's height plus the top (body underline) height.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import work_dir  # noqa: E402


def trace(verts, fx, fy, r, grow):
    axis = np.array([fx, fy], float)
    pts = []; radii = []
    z = 0.0; step = 0.02
    zmax = verts[:, 2].max()
    while z < zmax * 0.9:
        sl = verts[(verts[:, 2] >= z) & (verts[:, 2] < z + step)]
        if len(sl) == 0:
            z += step; continue
        d = np.linalg.norm(sl[:, :2] - axis, axis=1)
        near = sl[d < r]
        wide = sl[d < 2 * r]
        if len(near) < 3:
            break
        c = near[:, :2].mean(0)
        rad = np.percentile(np.linalg.norm(near[:, :2] - c, axis=1), 90)
        spread = np.percentile(np.linalg.norm(wide[:, :2] - c, axis=1), 90) if len(wide) else rad
        if len(radii) >= 4 and spread > grow * np.median(radii):
            break
        radii.append(rad)
        axis = 0.6 * axis + 0.4 * c
        pts.append((z + step / 2, axis[0], axis[1], rad))
        z += step
    return pts


def main():
    name = sys.argv[1]
    args = sys.argv[2:]
    r = float(args[args.index('--r') + 1]) if '--r' in args else 0.07
    grow = float(args[args.index('--grow') + 1]) if '--grow' in args else 2.2
    feet = [tuple(map(float, a.split(','))) for a in args if ',' in a]
    verts = np.load(work_dir(name, 'prep', 'verts.npy'))
    for fx, fy in feet:
        pts = trace(verts, fx, fy, r, grow)
        if not pts:
            print(f'foot ({fx:+.3f},{fy:+.3f}): no leg found'); continue
        top = pts[-1][0]
        out = []
        for f in (0, 0.25, 0.5, 0.75, 1.0):
            p = min(pts, key=lambda q: abs(q[0] - f * top))
            out.append(f'{int(f*100):3d}%:({p[1]:+.3f},{p[2]:+.3f},{p[0]:.3f})')
        med_r = np.median([p[3] for p in pts])
        print(f'foot ({fx:+.3f},{fy:+.3f}) top z={top:.3f} r~{med_r:.3f}  ' + ' '.join(out))


if __name__ == '__main__':
    main()
