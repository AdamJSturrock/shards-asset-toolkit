# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Visual fitting jig for critter rigs (workflow step 2).

python3 scripts/critter/fit_landmarks.py <critter> [--grid-only] [--out PNG]

Reads output/critter-authoring/<critter>/prep/{verts.npy,views.json,*.png} and
the critter's joints/bones from scripts/critter/profiles.json (or
$CRITTER_PROFILES), and
draws a 0.1-unit coordinate grid plus every joint and bone over the side,
front and top renders (rig space: head toward -Y, left = +X, feet at z = 0).
Left-side ('.L') joints are listed in the profile; '.R' joints are their
mirror in x. Also prints automatic foot-contact estimates (vertex clusters at
the very bottom of the mesh) as a starting point for leg landmarks.
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))

from paths import work_dir  # noqa: E402
OUT = work_dir()
from paths import profiles_path  # noqa: E402
PROFILES = profiles_path()

COLOURS = {
    'spine': (255, 220, 40), 'head': (255, 140, 0), 'tail': (180, 90, 255),
    'L': (255, 60, 60), 'R': (60, 140, 255), 'wing': (40, 220, 120), 'root': (255, 255, 255),
}


def project(view, p):
    c = np.array(view['center']); r = np.array(view['right']); u = np.array(view['up'])
    s = view['ortho_scale']; res = view['res']
    d = np.asarray(p, float) - c
    return ((d @ r) / s + 0.5) * res, (0.5 - (d @ u) / s) * res


def load_profile(name):
    from critter_templates import bones_for, full_joints
    prof = json.load(open(PROFILES))[name]
    joints = {k: np.array(v, float) for k, v in full_joints(prof['joints']).items()}
    bones = [(n, h, t) for n, h, t, _ in bones_for(prof['plan'], joints)]
    return prof, joints, bones


def bone_colour(name):
    if name.endswith('.L'):
        return COLOURS['wing'] if 'wing' in name else COLOURS['L']
    if name.endswith('.R'):
        return COLOURS['wing'] if 'wing' in name else COLOURS['R']
    for k in ('tail', 'head', 'root'):
        if name.startswith(k):
            return COLOURS[k]
    return COLOURS['spine']


def foot_contacts(verts, frac=0.035, rad_frac=0.06):
    z = verts[:, 2]
    low = verts[z < z.max() * frac]
    if len(low) == 0:
        return []
    # Greedy clustering on (x, y) with a radius of 6% of the body length
    # (narrower for close-set feet: a scorpion's, a crab's).
    rad = (verts[:, 1].max() - verts[:, 1].min()) * rad_frac
    pts = low[:, :2].copy(); left = np.ones(len(pts), bool); clusters = []
    while left.any():
        i = np.flatnonzero(left)[0]
        members = left & (np.linalg.norm(pts - pts[i], axis=1) < rad)
        # grow once so a cluster is the connected blob, not a disc
        for _ in range(6):
            cen = pts[members].mean(0)
            members = left & (np.linalg.norm(pts - cen, axis=1) < rad * 1.4)
        clusters.append((int(members.sum()), low[members].mean(0)))
        left &= ~members
    clusters.sort(key=lambda c: -c[0])
    return clusters


def label_tiles(prep, views, verts, joints, bones):
    """Second row, when fit_arthropod.py has written labels.npy: every
    vertex as a dot coloured by the limb that claims it (grey = body),
    nearest dots drawn last, with the bones on top. This is the ownership
    the rig's skin weights are confined to, so read it as a weight preview."""
    lp = os.path.join(prep, 'labels.npy')
    if not os.path.exists(lp):
        return []
    lab = np.load(lp)
    names = json.load(open(os.path.join(prep, 'labels.json')))['names']
    import colorsys
    pal = {i: tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 0.95)) for i in range(len(names))}
    pal[-1] = (150, 150, 150)
    out = []
    for vname in ('side', 'front', 'top'):
        view = views[vname]
        res = view['res']
        im = Image.new('RGB', (res, res), (255, 255, 255))
        d = ImageDraw.Draw(im)
        c = np.array(view['center']); r = np.array(view['right']); u = np.array(view['up'])
        depth = (verts - c) @ np.cross(r, u)       # toward the camera
        for i in np.argsort(depth):
            x, y = project(view, verts[i])
            d.point((x, y), fill=pal[int(lab[i])])
        for bname, h, t in bones:
            if h in joints and t in joints:
                d.line([project(view, joints[h]), project(view, joints[t])], fill=(0, 0, 0), width=2)
        d.text((6, 6), f'labels {vname}', fill=(0, 0, 0))
        yy = 20
        for i, n in enumerate(names):
            if vname == 'side':
                d.text((6, yy), n, fill=pal[i]); yy += 12
        out.append(im)
    return out


def zoom(name, spec, out_png):
    """--zoom view:u0,u1,v0,v1 : a close-up of one prep view over that window
    of its two in-plane axes (side: y,z; front: x,z; top: x,y) with a 0.05
    grid labelled every 0.1, for reading small hints (a fang tip, a claw's
    moving finger) off the mesh. Upscaled from the 768 px prep render."""
    vname, rng = spec.split(':')
    u0, u1, v0, v1 = (float(x) for x in rng.split(','))
    prep = os.path.join(OUT, name, 'prep')
    view = json.load(open(os.path.join(prep, 'views.json')))['views'][vname]
    im = Image.open(os.path.join(prep, f'{vname}.png')).convert('RGB')
    axes = {'side': (1, 2), 'front': (0, 2), 'top': (0, 1)}[vname]
    c = np.array(view['center'])

    def pt(u, v):
        p = c.copy(); p[axes[0]] = u; p[axes[1]] = v
        return project(view, p)
    xa, ya = pt(u0, v1); xb, yb = pt(u1, v0)
    box = (int(min(xa, xb)), int(min(ya, yb)), int(max(xa, xb)), int(max(ya, yb)))
    k = 900 / max(box[2] - box[0], box[3] - box[1])
    crop = im.crop(box).resize((int((box[2] - box[0]) * k), int((box[3] - box[1]) * k)))
    d = ImageDraw.Draw(crop)
    for ax_i, (lo, hi) in enumerate(((u0, u1), (v0, v1))):
        t = np.ceil(min(lo, hi) * 20) / 20
        while t <= max(lo, hi) + 1e-9:
            if ax_i == 0:
                x, _ = pt(t, v0); x = (x - box[0]) * k
                d.line([(x, 0), (x, crop.size[1])], fill=(90, 90, 90) if round(t * 20) % 2 == 0 else (170, 170, 170))
                if round(t * 20) % 2 == 0:
                    d.text((x + 2, 2), f'{"xyz"[axes[0]]}{t:+.1f}', fill=(0, 0, 160))
            else:
                _, y = pt(u0, t); y = (y - box[1]) * k
                d.line([(0, y), (crop.size[0], y)], fill=(90, 90, 90) if round(t * 20) % 2 == 0 else (170, 170, 170))
                if round(t * 20) % 2 == 0:
                    d.text((2, y + 2), f'{"xyz"[axes[1]]}{t:+.1f}', fill=(0, 0, 160))
            t += 0.05
    crop.save(out_png)
    print('wrote', out_png)


def main():
    name = sys.argv[1]
    if '--zoom' in sys.argv:
        spec = sys.argv[sys.argv.index('--zoom') + 1]
        zoom(name, spec, os.path.join(OUT, name, f'zoom_{spec.split(":")[0]}.png'))
        return
    grid_only = '--grid-only' in sys.argv
    out_png = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(OUT, name, 'fit.png')
    prep = os.path.join(OUT, name, 'prep')
    views = json.load(open(os.path.join(prep, 'views.json')))['views']
    verts = np.load(os.path.join(prep, 'verts.npy'))
    joints, bones = {}, []
    if not grid_only:
        _, joints, bones = load_profile(name)
    tiles = []
    for vname in ('side', 'front', 'top'):
        view = views[vname]
        im = Image.open(os.path.join(prep, f'{vname}.png')).convert('RGB')
        d = ImageDraw.Draw(im)
        # grid: 0.1 units, labelled every 0.2
        axes = {'side': (1, 2), 'front': (0, 2), 'top': (0, 1)}[vname]
        lo = np.floor(verts.min(0) * 10) / 10 - 0.1
        hi = np.ceil(verts.max(0) * 10) / 10 + 0.1
        for ax_i, ax in enumerate(axes):
            for t in np.arange(lo[ax], hi[ax] + 1e-6, 0.1):
                p0 = np.zeros(3); p1 = np.zeros(3)
                other = axes[1 - ax_i]
                p0[ax] = p1[ax] = t; p0[other] = lo[other]; p1[other] = hi[other]
                a, b = project(view, p0), project(view, p1)
                major = abs(round(t * 10)) % 2 == 0
                d.line([a, b], fill=(130, 130, 130) if major else (175, 175, 175), width=1)
                if major:
                    lbl = f'{"xyz"[ax]}{t:+.1f}'
                    d.text((a[0] + 2, a[1] + 2) if ax_i == 0 else (a[0] + 2, a[1] - 12), lbl, fill=(60, 60, 60))
        for bname, h, t in bones:
            if h in joints and t in joints:
                d.line([project(view, joints[h]), project(view, joints[t])], fill=bone_colour(bname), width=4)
        for jn, p in joints.items():
            x, y = project(view, p)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 0, 0), fill=(255, 255, 255))
            if not jn.endswith('.R') or vname == 'front':
                d.text((x + 5, y - 5), jn, fill=(0, 0, 0))
        d.text((6, 6), f'{name} {vname}', fill=(0, 0, 0))
        tiles.append(im)
    lab_tiles = label_tiles(prep, views, verts, joints, bones)
    W = sum(t.size[0] for t in tiles); H = max(t.size[1] for t in tiles) * (2 if lab_tiles else 1)
    sheet = Image.new('RGB', (W, H), (255, 255, 255)); x = 0
    for t in tiles:
        sheet.paste(t, (x, 0)); x += t.size[0]
    x = 0
    for t in lab_tiles:
        sheet.paste(t, (x, H // 2)); x += t.size[0]
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sheet.save(out_png)
    print('wrote', out_png)
    for n, c in foot_contacts(verts)[:6]:
        print(f'  foot-contact cluster n={n:4d} at x={c[0]:+.3f} y={c[1]:+.3f} z={c[2]:.3f}')


if __name__ == '__main__':
    main()
