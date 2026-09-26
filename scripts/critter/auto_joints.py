# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Derive a quadruped skeleton's leg and spine joints from the mesh itself.

python3 scripts/critter/auto_joints.py <critter> --write

Inputs, all in rig space (head toward -Y, left = +X, feet at z = 0):
  - the four foot contacts (lowest-vertex clusters; profile `feet` overrides)
  - the body underline and topline at the front and hind legs (hand-read
    `lines` in the profile, else measured on the midline)
Joints are placed at fixed anatomical fractions of those measurements:
  front leg: shoulder 45% up the body, elbow at the underline, fetlock at
             30% of the underline height, toe on the ground ahead of the foot
  hind leg:  hip 50% up the body, stifle at the underline, hock at
             `hock_frac` (default 0.42) of the underline height, toe ahead
  spine:     hips over the hind legs, chest over the front legs, 65% up
The head, neck and tail joints are anatomy that no formula places well;
they come from the profile's hand-read `head`, `tail` lists. With --write
the result is merged into profiles.json under `joints` (existing hand
overrides in `joint_overrides` win). Every leg is fitted on its own side:
Meshy poses are often asymmetric, so nothing is mirrored.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from fit_landmarks import foot_contacts  # noqa: E402
from paths import work_dir, profiles_path  # noqa: E402
PROFILES = profiles_path()


def body_lines(v, y, x0, half_w=0.06):
    """Underline/topline z of the BODY at y, measured in a column at x0
    (the leg's own x, so neighbouring legs at the midline don't count)."""
    s = v[(np.abs(v[:, 1] - y) < half_w) & (np.abs(v[:, 0] - x0) < 0.06)]
    s = s[s[:, 2] > 0.12 * v[:, 2].max()]  # ignore the leg/hoof itself
    return float(s[:, 2].min()), float(s[:, 2].max())


def main():
    name = sys.argv[1]
    profs = json.load(open(PROFILES))
    prof = profs[name]
    v = np.load(work_dir(name, 'prep', 'verts.npy'))
    L = v[:, 1].max() - v[:, 1].min()
    if 'feet' in prof:
        feet = {k: np.array(p, float) for k, p in prof['feet'].items()}
    else:
        cl = [c for _, c in foot_contacts(v)[:4]]
        cl.sort(key=lambda c: c[1])
        front, hind = sorted(cl[:2], key=lambda c: -c[0]), sorted(cl[2:], key=lambda c: -c[0])
        feet = {'front.L': front[0], 'front.R': front[1], 'hind.L': hind[0], 'hind.R': hind[1]}
    hock_frac = prof.get('hock_frac', 0.42)
    J = {}
    lines = {}
    for leg, f in feet.items():
        fx, fy = float(f[0]), float(f[1])
        side = leg.split('.')[1]
        # Body underline/topline over this leg: hand-read from the grid when
        # the profile gives `lines` (a leg's inner surface or an udder spoils
        # any automatic measurement), else measured on the midline.
        if 'lines' in prof:
            u, t = prof['lines']['front' if leg.startswith('front') else 'hind']
        else:
            u, t = body_lines(v, fy, 0.0)
        lines[leg] = (u, t)
        if leg.startswith('front'):
            J[f'shoulder.{side}'] = [fx * 0.85, fy, u + 0.45 * (t - u)]
            J[f'elbow.{side}'] = [fx, fy + 0.01, u]
            J[f'fetlock.{side}'] = [fx, fy, 0.30 * u]
            J[f'ftoe.{side}'] = [fx, fy - 0.06 * L, 0.0]
        else:
            J[f'hip.{side}'] = [fx * 0.85, fy, u + 0.50 * (t - u)]
            J[f'stifle.{side}'] = [fx, fy - 0.02, u]
            J[f'hock.{side}'] = [fx, fy + 0.03, hock_frac * u]
            J[f'htoe.{side}'] = [fx, fy - 0.05 * L, 0.0]
    fy_front = np.mean([feet['front.L'][1], feet['front.R'][1]])
    fy_hind = np.mean([feet['hind.L'][1], feet['hind.R'][1]])
    uf, tf = prof['lines']['front'] if 'lines' in prof else body_lines(v, fy_front, 0.0)
    uh, th = prof['lines']['hind'] if 'lines' in prof else body_lines(v, fy_hind, 0.0)
    # The spine runs over the MIDDLE of each foot pair, not x = 0: a mesh whose
    # tail skews its bounding box (the rat) has its body off the bbox centre.
    fx_front = float(np.mean([feet['front.L'][0], feet['front.R'][0]]))
    fx_hind = float(np.mean([feet['hind.L'][0], feet['hind.R'][0]]))
    J['chest'] = [fx_front, fy_front, uf + 0.65 * (tf - uf)]
    J['hips'] = [fx_hind, fy_hind, uh + 0.65 * (th - uh)]
    J['spine'] = [(fx_front + fx_hind) / 2, (fy_front + fy_hind) / 2, (J['chest'][2] + J['hips'][2]) / 2]
    # Ground-level root under the body centre, pointing forward: the whole
    # critter pivots on it (death roll, hop lift) without touching the spine.
    J['root'] = [J['spine'][0], J['spine'][1], 0.0]
    J['root_tip'] = [J['spine'][0], J['spine'][1] - 0.15 * L, 0.0]
    for k, p in prof.get('head', {}).items():
        J[k] = p
    for k, p in prof.get('tail', {}).items():
        J[k] = p
    for k, p in prof.get('joint_overrides', {}).items():
        J[k] = p
    J = {k: [round(float(c), 3) for c in p] for k, p in J.items()}
    print(json.dumps({'critter': name, 'lines': {k: [round(a, 3), round(b, 3)] for k, (a, b) in lines.items()},
                      'feet': {k: [round(float(c), 3) for c in f[:2]] for k, f in feet.items()}}))
    if '--write' in sys.argv:
        prof['joints'] = J
        profs[name] = prof
        json.dump(profs, open(PROFILES, 'w'), indent=1)
        print('wrote joints for', name)


if __name__ == '__main__':
    main()
