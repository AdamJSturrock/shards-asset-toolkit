# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Render the limb segmentation (fit_arthropod.py's labels) BEFORE rigging.

python3 scripts/critter/render_labels.py <critter>

Writes <work>/<critter>/labels.png: the mesh with every appendage label in
its own flat colour, from top, underside, front, side and 3/4, with a legend
of label names and vertex counts, and the label checks (fit-report.json)
printed underneath. Colours are fixed by label NAME, so the same limb has the
same colour on every creature:
  legs     one hue per leg number (front to back); left light, right dark
  claw     magenta, its moving finger (dactyl) pink
  fang     blue      palp  violet     eye  gold
  tail     purple    head/neck  brown wing  teal
  body     neutral grey
Runs Blender (headless) on itself for the renders, then composes with Pillow.
"""
import colorsys, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import work_dir  # noqa: E402

LEG_HUES = [0.0, 0.085, 0.16, 0.33, 0.5, 0.58, 0.95]
KIND_HUE = {'claw': 0.83, 'dactyl': 0.9, 'fang': 0.64, 'palp': 0.74, 'eye': 0.12, 'tail': 0.78,
            'head': 0.06, 'wing': 0.45}
VIEWS = ('top', 'underside', 'front', 'side', 'q34')


def colour(name):
    """RGB 0..1 for a label name; None (body) -> grey."""
    if name is None:
        return (0.62, 0.62, 0.62)
    kind, _, side = name.partition('.')
    dark = side == 'R'
    if kind.startswith('leg'):
        h = LEG_HUES[(int(kind[3:]) - 1) % len(LEG_HUES)]
    else:
        h = KIND_HUE.get(kind, 0.3)
    s, v = (0.85, 1.0) if not dark else (0.95, 0.55)
    if kind == 'head':
        s, v = 0.65, 0.6
    return colorsys.hsv_to_rgb(h, s, v)


def blender_part(name, tiles_dir, size):
    import bpy
    import numpy as np
    from mathutils import Vector, Matrix
    prep = work_dir(name, 'prep')
    v = np.load(os.path.join(prep, 'verts.npy'))
    f = np.load(os.path.join(prep, 'faces.npy'))
    lab = np.load(os.path.join(prep, 'labels.npy'))
    names = json.load(open(os.path.join(prep, 'labels.json')))['names']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    mats = {}
    for i in [-1] + list(range(len(names))):
        m = bpy.data.materials.new(f'L{i}')
        c = colour(None if i < 0 else names[i])
        m.diffuse_color = (*c, 1.0)
        mats[i] = len(me.materials)
        me.materials.append(m)
    # a face takes the label most of its corners carry
    fl = lab[f]
    maj = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
    me.polygons.foreach_set('material_index', [mats[int(x)] for x in maj])
    me.update()
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_cavity = True
    sc.display.render_aa = '8'
    sc.render.resolution_x = sc.render.resolution_y = size
    w = bpy.data.worlds.new('W'); sc.world = w; w.color = (1, 1, 1)
    sc.view_settings.view_transform = 'Standard'
    cd = bpy.data.cameras.new('C'); cd.type = 'ORTHO'
    cam = bpy.data.objects.new('C', cd); sc.collection.objects.link(cam); sc.camera = cam
    mn, mx = v.min(0), v.max(0)
    C = Vector(((mn + mx) / 2).tolist())
    R = float((mx - mn).max())
    dirs = {'top': (Vector((0, 0, 1)), Vector((0, -1, 0))), 'underside': (Vector((0, 0, -1)), Vector((0, -1, 0))),
            'front': (Vector((0, -1, 0)), Vector((0, 0, 1))), 'side': (Vector((1, 0, 0)), Vector((0, 0, 1))),
            'q34': (Vector((1, -1, 0.9)).normalized(), Vector((0, 0, 1)))}
    for vn in VIEWS:
        d, up = dirs[vn]
        cam.location = C + d * R * 4
        look = -d
        right = look.cross(up).normalized(); tu = right.cross(look)
        cam.rotation_euler = Matrix((right, tu, -look)).transposed().to_euler()
        cd.ortho_scale = R * 1.15
        cd.clip_start, cd.clip_end = 0.001, R * 20
        sc.render.filepath = os.path.join(tiles_dir, f'{vn}.png')
        bpy.ops.render.render(write_still=True)


def compose(name, tiles_dir, size):
    from PIL import Image, ImageDraw
    import numpy as np
    prep = work_dir(name, 'prep')
    names = json.load(open(os.path.join(prep, 'labels.json')))['names']
    lab = np.load(os.path.join(prep, 'labels.npy'))
    rep = json.load(open(os.path.join(prep, 'fit-report.json')))
    checks = rep.get('checks', {})
    leg_w = 300
    sheet = Image.new('RGB', (size * 3 + leg_w, size * 2 + 150), (255, 255, 255))
    for k, vn in enumerate(VIEWS):
        im = Image.open(os.path.join(tiles_dir, f'{vn}.png')).convert('RGB')
        x, y = (k % 3) * size, (k // 3) * size
        sheet.paste(im, (x, y))
        ImageDraw.Draw(sheet).text((x + 6, y + 6), vn, fill=(0, 0, 0))
    d = ImageDraw.Draw(sheet)
    x0 = size * 3 + 10
    d.text((x0, 8), f'{name}: {len(names)} limb labels', fill=(0, 0, 0))
    yy = 30
    for i, n in [(-1, 'body')] + list(enumerate(names)):
        c = tuple(int(255 * t) for t in colour(None if i < 0 else n))
        d.rectangle([x0, yy, x0 + 16, yy + 12], fill=c, outline=(0, 0, 0))
        d.text((x0 + 24, yy), f'{n}  {int((lab == i).sum())} verts', fill=(0, 0, 0))
        yy += 17
    # the checks, under the renders
    yy = size * 2 + 8
    ok = checks.get('pass')
    d.text((8, yy), 'LABEL CHECKS: ' + ('PASS' if ok else 'FAIL'), fill=(0, 128, 0) if ok else (200, 0, 0))
    yy += 16
    for line in (checks.get('failures') or [])[:6]:
        d.text((8, yy), line, fill=(200, 0, 0)); yy += 14
    for line in (checks.get('accepted') or [])[:4]:
        d.text((8, yy), line, fill=(200, 110, 0)); yy += 14
    mir = checks.get('b_mirror', {})
    d.text((8, yy), 'mirror L/R: ' + '  '.join(f'{k} {a}/{b}' for k, (a, b, _) in mir.items()), fill=(0, 0, 0))
    yy += 14
    d.text((8, yy), f"cleanup: {rep.get('label_cleanup', {})}", fill=(0, 0, 0))
    out = work_dir(name, 'labels.png')
    sheet.save(out)
    return out


if __name__ == '__main__':
    if '--' in sys.argv:        # inside Blender
        a = sys.argv[sys.argv.index('--') + 1:]
        blender_part(a[0], a[1], int(a[2]))
    else:
        name = sys.argv[1]
        size = 420
        tiles = work_dir(name, '.label_tiles')
        os.makedirs(tiles, exist_ok=True)
        # BLENDER_PATH (as in the toolkit's .env) wins over `blender` on PATH
        blender = os.environ.get('BLENDER_PATH') or 'blender'
        r = subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                            os.path.abspath(__file__), '--', name, tiles, str(size)],
                           capture_output=True, text=True, env=dict(os.environ))
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit('label render failed')
        print('wrote', compose(name, tiles, size))
