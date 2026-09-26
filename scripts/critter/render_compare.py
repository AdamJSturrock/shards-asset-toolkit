# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Before/after render of two meshes (a limb transplant, a mesh fix).

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/render_compare.py -- \
  --a <before.glb> --b <after.glb> --out <compare.png> [--labels "before,after"]

Textured Workbench renders of both, each normalised as prep does (bbox
centred in x/y, lowest point at z = 0), from the top (head up) and from the
RTS 3/4 camera, with the SAME camera framing for both so the change is the
only difference. Output: a 2 x 2 sheet, before on the left.
"""
import bpy, sys, os
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
SIZE = int(opt.get('size', 640))
labels = opt.get('labels', 'before,after').split(',')


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.expanduser(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    pts = np.array([v.co[:] for o in meshes for v in o.data.vertices])
    mn, mx = pts.min(0), pts.max(0)
    off = Vector((-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]))
    for o in meshes:
        o.location = off
    return pts + np.array(off)


def setup(extent_pts):
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'; sh.color_type = 'TEXTURE'; sh.show_cavity = True
    sc.display.render_aa = '8'
    sc.render.resolution_x = sc.render.resolution_y = SIZE
    w = bpy.data.worlds.new('W'); sc.world = w; w.color = (0.62, 0.64, 0.6)
    sc.view_settings.view_transform = 'Standard'
    cd = bpy.data.cameras.new('C'); cd.type = 'ORTHO'
    cam = bpy.data.objects.new('C', cd); sc.collection.objects.link(cam); sc.camera = cam
    return cam, cd


def shoot(cam, cd, C, R, view, path):
    d, up = {'top': (Vector((0, 0, 1)), Vector((0, -1, 0))),
             'rts': (Vector((-0.55, -0.55, 0.9)).normalized(), Vector((0, 0, 1)))}[view]
    cam.location = C + d * R * 4
    look = -d
    right = look.cross(up).normalized(); tu = right.cross(look)
    cam.rotation_euler = Matrix((right, tu, -look)).transposed().to_euler()
    cd.ortho_scale = R * 1.2
    cd.clip_start, cd.clip_end = 0.001, R * 20
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


out = os.path.abspath(os.path.expanduser(opt['out']))
tmp = out + '.tiles'
os.makedirs(tmp, exist_ok=True)
pa = load(opt['a'])
C = Vector(((pa.min(0) + pa.max(0)) / 2).tolist())
R = float(np.ptp(pa, axis=0).max()) * 1.1
for tag, path in (('a', opt['a']), ('b', opt['b'])):
    load(path)
    cam, cd = setup(None)
    for view in ('top', 'rts'):
        shoot(cam, cd, C, R, view, os.path.join(tmp, f'{tag}_{view}.png'))
# compose (Blender's image API, so no Pillow is needed inside Blender)
W = SIZE
sheet = np.zeros((2 * W, 2 * W, 4), dtype=np.float32)
for c, tag in enumerate(('a', 'b')):
    for r, view in enumerate(('top', 'rts')):
        im = bpy.data.images.load(os.path.join(tmp, f'{tag}_{view}.png'))
        px = np.array(im.pixels[:], dtype=np.float32).reshape(W, W, 4)
        sheet[(1 - r) * W:(2 - r) * W, c * W:(c + 1) * W] = px
img = bpy.data.images.new('cmp', 2 * W, 2 * W, alpha=True)
img.pixels = sheet.ravel()
img.filepath_raw = out; img.file_format = 'PNG'; img.save()
print('COMPARE', out, labels)
