# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Render review frames for every animation clip in a skinned GLB.

For each action: evenly sampled frames from a game-like three-quarter overhead
camera and from the side, with fixed framing across clips so poses compare.
`scripts/clip_review_sheet.py` turns the frames into contact sheets, including
a strip downscaled to the unit's on-screen height (gameplay-size
review).

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/clip_review.py -- \
      --input output/anim-authoring/goblin_worker/v1/candidates.glb \
      --output-dir output/anim-authoring/goblin_worker/v1/review --samples 8

Frames land in <output-dir>/<clip>/<view>-NN.png, with frames.json listing
the sampled frame numbers and seconds. The output directory must be empty.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def held_tools(rig, meshes, height, specs):
    """Parent tool GLBs (grip at origin, handle +Y, head dir +Z in glTF space) to RightHand.

    Grip sits at the palm: the centroid of vertices mostly bound to RightHand at
    rest. At rest the handle points forward (-Y in Blender) and the head's
    working edge faces down, which reads as a held tool once the arm swings.
    Returns {clip: holder_object}; holders are hidden except on their clip.
    """
    from mathutils import Matrix
    holders = {}
    mesh = meshes[0]
    group = mesh.vertex_groups.get('RightHand')
    if not specs or group is None:
        return holders
    pts = [mesh.matrix_world @ v.co for v in mesh.data.vertices
           if any(g.group == group.index and g.weight > 0.7 for g in v.groups)]
    palm = sum(pts, Vector()) / len(pts)
    # Columns: tool X -> world X, tool up (Blender +Y after import, i.e. -headDir) -> world up,
    # tool handle (Blender +Z) -> world forward (-Y).
    rot = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))).transposed().to_4x4()
    for spec in specs:
        clip, rest = spec.split('=', 1)
        path, _, frac = rest.partition('@')
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(Path(path).resolve()))
        new = [o for o in bpy.data.objects if o not in before]
        holder = bpy.data.objects.new(f'held_{clip}', None)
        bpy.context.scene.collection.objects.link(holder)
        for o in new:
            if o.parent is None:
                o.parent = holder
        tool_len = max((o.matrix_world @ Vector(c)).z for o in new if o.type == 'MESH' for c in o.bound_box) - \
            min((o.matrix_world @ Vector(c)).z for o in new if o.type == 'MESH' for c in o.bound_box)
        scale = height * float(frac or 0.5) / max(tool_len, 1e-6)
        holder.parent = rig
        holder.parent_type = 'BONE'
        holder.parent_bone = 'RightHand'
        bpy.context.view_layer.update()
        holder.matrix_world = Matrix.Translation(palm) @ rot @ Matrix.Scale(scale, 4)
        holders[clip] = (holder, new)
    return holders


def show_tools(tools, clip):
    for name, (holder, objs) in tools.items():
        for o in objs:
            o.hide_render = name != clip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples', type=int, default=8, help='0 renders every frame (for review video)')
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--clips', default='', help='comma-separated subset of clip names')
    parser.add_argument('--views', default='game,side')
    parser.add_argument('--held-tool', action='append', default=[],
                        help='clip=path/to/tool.glb[@heightFraction]; shows the tool in the right hand for that clip')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use an empty review directory to preserve prior evidence')

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Before import: the glTF importer converts clip seconds to frames at the scene
    # rate, and the default 24 fps would make 30 fps review video play 25% fast.
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(Path(args.input).resolve()))
    for o in list(bpy.context.scene.objects):
        if o.type == 'MESH' and not any(m.type == 'ARMATURE' for m in o.modifiers):
            bpy.data.objects.remove(o)
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    height = max((o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box)
    if rig.animation_data:
        rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    tools = held_tools(rig, meshes, height, args.held_tool)

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    world = bpy.data.worlds.new('review')
    world.color = (0.33, 0.36, 0.33)
    scene.world = world
    bpy.ops.object.light_add(type='SUN', rotation=(math.radians(45), 0, math.radians(35)))
    bpy.context.object.data.energy = 3.5
    bpy.ops.mesh.primitive_plane_add(size=40)
    ground = bpy.context.object
    mat = bpy.data.materials.new('ground')
    mat.diffuse_color = (0.22, 0.26, 0.2, 1)
    ground.data.materials.append(mat)
    cam_data = bpy.data.cameras.new('review')
    cam_data.type = 'ORTHO'
    # frame on the larger of height and footprint: a humanoid is framed by
    # its height as before, a spider or a crab (wide and low) by its legs
    reach = max(max(abs((o.matrix_world @ Vector(c))[k]) for o in meshes for c in o.bound_box) for k in (0, 1))
    size = max(height, 1.1 * reach)
    cam_data.ortho_scale = size * 1.9
    cam = bpy.data.objects.new('review', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    target = Vector((0, 0, height * 0.5))
    views = {
        # RTS camera: looking down ~50 degrees from the front-left quarter.
        'game': target + Vector((-0.55, -0.55, 0.9)).normalized() * size * 6,
        'side': target + Vector((1, 0, 0)) * size * 6,
        'front': target + Vector((0, -1, 0)) * size * 6,
    }
    wanted = [c for c in args.clips.split(',') if c]
    actions = [a for a in bpy.data.actions if not wanted or a.name in wanted]
    report = {}
    rig.animation_data_create()
    for action in actions:
        show_tools(tools, action.name)
        rig.animation_data.action = action
        if action.slots:
            rig.animation_data.action_slot = action.slots[0]
        start, end = action.frame_range
        if args.samples <= 0:  # every frame, for playback video
            frames = [float(f) for f in range(int(start), int(end) + 1)]
        else:
            frames = [start + (end - start) * i / max(args.samples - 1, 1) for i in range(args.samples)]
        clip_dir = out / action.name
        clip_dir.mkdir()
        for view in args.views.split(','):
            cam.location = views[view]
            cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
            for i, f in enumerate(frames):
                scene.frame_set(int(f), subframe=f - int(f))
                scene.render.filepath = str(clip_dir / f'{view}-{i:04d}.png')  # 4 digits: 2 sorted game-10 before game-100 and scrambled long clips
                bpy.ops.render.render(write_still=True)
        report[action.name] = {'frames': [round(f, 2) for f in frames],
                               'seconds': [round((f - start) / 30, 3) for f in frames],
                               'duration': round((end - start) / 30, 3)}
    (out / 'frames.json').write_text(json.dumps({'input': args.input, 'clips': report}, indent=2) + '\n')


if __name__ == '__main__':
    main()
