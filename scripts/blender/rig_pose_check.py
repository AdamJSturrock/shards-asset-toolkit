# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Render a skinned humanoid GLB's weights and a set of stress poses, for review.

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/rig_pose_check.py -- \
      --input output/anim-authoring/<id>/rig-v1/character.glb --out <dir>/rigcheck.png

One sheet. Row 1: dominant-bone colours (front, side, back, 3/4). Rows 2-3:
textured test poses from the front and the 3/4 game view: rest; arms raised
forward and out; a stride (one thigh forward, knee bent); a forward spine
bend; elbows bent 90 degrees. Poses are world-axis rotations applied down
the Meshy 24-joint chain, so they read the same on any rig with those names.
Numbers cannot show a shoulder tearing through a carapace: look at the sheet.
"""
import argparse
import colorsys
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
POSES = {
    'rest': [],
    'arms': [('LeftArm', Y, -60), ('RightArm', Y, 60), ('LeftArm', X, 50), ('RightArm', X, 50)],
    'armsdown': [('LeftArm', Y, 70), ('RightArm', Y, -70), ('LeftForeArm', X, 60), ('RightForeArm', X, 60)],
    'stride': [('LeftUpLeg', X, 35), ('LeftLeg', X, -50), ('RightUpLeg', X, -25), ('RightLeg', X, -20)],
    'bend': [('Spine02', X, 15), ('Spine01', X, 15), ('Spine', X, 10), ('neck', X, 5)],
    'elbows': [('LeftForeArm', Z, 80), ('RightForeArm', Z, -80), ('LeftForeArm', X, 30), ('RightForeArm', X, 30)],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--size', type=int, default=320)
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(Path(args.input).resolve()))
    for o in list(sc.objects):
        if o.type == 'MESH' and not any(m.type == 'ARMATURE' for m in o.modifiers):
            bpy.data.objects.remove(o)
    rig = next(o for o in sc.objects if o.type == 'ARMATURE')
    mesh = next(o for o in sc.objects if o.type == 'MESH')
    if rig.animation_data:
        rig.animation_data.action = None
    rest = {b.name: (rig.matrix_world @ b.matrix_local).to_quaternion() for b in rig.data.bones}

    # dominant-bone colour attribute
    names = sorted(g.name for g in mesh.vertex_groups)
    hue = {n: (i * 0.618) % 1.0 for i, n in enumerate(names)}
    gname = {g.index: g.name for g in mesh.vertex_groups}
    me = mesh.data
    attr = me.color_attributes.new('dominant', 'FLOAT_COLOR', 'POINT')
    for v in me.vertices:
        c = np.zeros(3)
        for g in v.groups:
            c += np.array(colorsys.hsv_to_rgb(hue[gname[g.group]], 0.85, 0.95)) * g.weight
        attr.data[v.index].color = (*c, 1)

    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'
    sc.render.resolution_x = sc.render.resolution_y = args.size
    sc.view_settings.view_transform = 'Standard'
    w = bpy.data.worlds.new('W')
    sc.world = w
    w.color = (0.62, 0.64, 0.6)
    cd = bpy.data.cameras.new('C')
    cd.type = 'ORTHO'
    cam = bpy.data.objects.new('C', cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    co = np.array([mesh.matrix_world @ v.co for v in me.vertices])
    H = float(co[:, 2].max())
    C = Vector((0, 0, H * 0.5))
    R = float(np.ptp(co, axis=0).max())

    def aim(d):
        d = Vector(d).normalized()
        cam.location = C + d * R * 5
        look = -d
        right = look.cross(Z).normalized()
        up = right.cross(look)
        cam.rotation_euler = Matrix((right, up, -look)).transposed().to_euler()
        cd.ortho_scale = R * 1.5
        cd.clip_end = R * 20

    tiles = []
    out = Path(args.out).resolve()
    tmp = out.parent / (out.stem + '_tiles')
    tmp.mkdir(parents=True, exist_ok=True)

    def shot(tag):
        path = tmp / f'{len(tiles):02d}_{tag}.png'
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        tiles.append(path)

    sh.color_type = 'VERTEX'
    for tag, d in (('front', (0, -1, 0)), ('side', (1, 0, 0)), ('back', (0, 1, 0)), ('q34', (-1, -1, 0.9))):
        aim(d)
        shot('w_' + tag)
    sh.color_type = 'TEXTURE'
    pb = rig.pose.bones
    for view, d in (('front', (0, -1, 0.15)), ('q34', (-1, -1, 0.9))):
        for pose, rots in POSES.items():
            for b in pb:
                b.rotation_mode = 'QUATERNION'
                b.rotation_quaternion = Quaternion()
            for bone, axis, deg in rots:
                r = rest[bone]
                local = r.inverted() @ Quaternion(axis, math.radians(deg)) @ r
                pb[bone].rotation_quaternion = local @ pb[bone].rotation_quaternion
            bpy.context.view_layer.update()
            aim(d)
            shot(f'{view}_{pose}')
    cols = max(4, len(POSES))
    rows = 1 + 2
    S = args.size
    sheet = np.full((rows * S, cols * S, 4), 0.62, dtype=np.float32)
    sheet[..., 3] = 1
    layout = [tiles[:4], tiles[4:4 + len(POSES)], tiles[4 + len(POSES):]]
    for r, row in enumerate(layout):
        for c, path in enumerate(row):
            im = bpy.data.images.load(str(path))
            a = np.array(im.pixels[:], dtype=np.float32).reshape(S, S, 4)
            sheet[(rows - 1 - r) * S:(rows - r) * S, c * S:(c + 1) * S] = a
            bpy.data.images.remove(im)
    img = bpy.data.images.new('sheet', cols * S, rows * S, alpha=True)
    img.pixels = sheet.ravel()
    img.filepath_raw = str(out)
    img.file_format = 'PNG'
    img.save()
    for path in tiles:
        path.unlink()
    tmp.rmdir()
    print('RIGCHECK', out, 'columns:', ', '.join(POSES))


if __name__ == '__main__':
    main()
