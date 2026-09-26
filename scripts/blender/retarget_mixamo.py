# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Retarget Mixamo FBX clips onto a Meshy humanoid GLB skeleton.

Mixamo's rig and Meshy's 24-joint rig share a topology but not bone rolls,
rest poses or proportions. Each mapped bone receives the source bone's
world-space rotation *delta from its rest pose*. Limb bones are first aligned
so the target's rest direction matches the source's (a T-pose arm on Mixamo
drives a slightly lowered Meshy arm correctly). `--leg-align` below 1 keeps
part of the character's own stance, e.g. a goblin's bowed legs.
Head, feet and toes take the delta only; their bone axes are rig convention,
not anatomy, so aligning them would tilt the head.

Hips translation is scaled by hip height. Loops listed in `--loop` get any
horizontal drift removed so they stay in place. Clips listed in `--mirror` are
mirrored left-right (Left* bones take the Right* source motion reflected
across the character's YZ plane), e.g. a right-handed sword slash for a unit
that holds its sword in the left hand.

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/retarget_mixamo.py -- \
      --target input/goblin_worker/meshy_rigged.glb \
      --clip idle_crouch=mocap/goblin_worker/crouch_idle.fbx ... \
      --loop idle_crouch --out output/anim-authoring/goblin_worker/v1/candidates.glb

Writes `<out>.json` beside the GLB with the clip table (source, frames,
seconds). The retarget is a starting point for per-unit polish in Blender;
review it with scripts/blender/clip_review.py before promoting anything.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

MESHY_TO_MIXAMO = {
    'Hips': 'Hips', 'Spine02': 'Spine', 'Spine01': 'Spine1', 'Spine': 'Spine2',
    'neck': 'Neck', 'Head': 'Head',
    'LeftShoulder': 'LeftShoulder', 'LeftArm': 'LeftArm', 'LeftForeArm': 'LeftForeArm', 'LeftHand': 'LeftHand',
    'RightShoulder': 'RightShoulder', 'RightArm': 'RightArm', 'RightForeArm': 'RightForeArm', 'RightHand': 'RightHand',
    'LeftUpLeg': 'LeftUpLeg', 'LeftLeg': 'LeftLeg', 'LeftFoot': 'LeftFoot', 'LeftToeBase': 'LeftToeBase',
    'RightUpLeg': 'RightUpLeg', 'RightLeg': 'RightLeg', 'RightFoot': 'RightFoot', 'RightToeBase': 'RightToeBase',
}
ARM_ALIGN = {'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
             'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
             'Spine02', 'Spine01', 'Spine', 'neck'}
LEG_ALIGN = {'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', required=True, help='Meshy character.glb (skinned)')
    parser.add_argument('--clip', action='append', required=True, help='name=path/to/mixamo.fbx')
    parser.add_argument('--loop', action='append', default=[], help='clip names to keep in place')
    parser.add_argument('--mirror', action='append', default=[], help='clip names to mirror left-right')
    parser.add_argument('--leg-align', type=float, default=0.5, help='0 keeps own leg stance, 1 copies Mixamo stance')
    parser.add_argument('--keep-existing', action='store_true', help='also export the GLB\'s existing actions')
    parser.add_argument('--out', required=True)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def rot(m):
    return m.to_3x3().normalized().to_quaternion()


def armature(objs):
    return next(o for o in objs if o.type == 'ARMATURE')


def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    new = [o for o in bpy.data.objects if o not in before]
    src = armature(new)
    action = src.animation_data.action if src.animation_data else None
    if action is None:
        raise RuntimeError(f'{path}: no animation on imported armature')
    return src, action, new


def bone_name(src, mixamo):
    for candidate in (f'mixamorig:{mixamo}', mixamo, f'mixamorig1:{mixamo}'):
        if candidate in src.pose.bones:
            return candidate
    raise RuntimeError(f'Mixamo bone {mixamo} not found')


def order(tgt):
    out = []

    def walk(b):
        out.append(b.name)
        for c in b.children:
            walk(c)
    for b in tgt.data.bones:
        if b.parent is None:
            walk(b)
    return out


def swap_side(name):
    if name.startswith('Left'):
        return 'Right' + name[4:]
    if name.startswith('Right'):
        return 'Left' + name[5:]
    return name


def retarget(tgt, src, action, name, leg_align, in_place, mirror=False):
    scene = bpy.context.scene
    src.animation_data.action = action
    start, end = (int(round(v)) for v in action.frame_range)
    names = {t: bone_name(src, swap_side(m) if mirror else m)
             for t, m in MESHY_TO_MIXAMO.items() if t in tgt.pose.bones}
    tw, sw = tgt.matrix_world, src.matrix_world
    # Mirroring across the YZ plane: a rotation (w, x, y, z) becomes (w, x, -y, -z)
    # (axial vectors flip every component but x), a position or direction negates x.
    mq = (lambda q: Quaternion((q.w, q.x, -q.y, -q.z))) if mirror else (lambda q: q)
    mv = (lambda v: Vector((-v.x, v.y, v.z))) if mirror else (lambda v: v)
    tgt_rest = {b.name: rot(tw @ b.matrix_local) for b in tgt.data.bones}
    scene.frame_set(start)
    # Mixamo exports its rest pose as the armature rest (T-pose); read it from bones, not frame 1.
    src_rest = {t: mq(rot(sw @ src.data.bones[s].matrix_local)) for t, s in names.items()}
    align = {}
    for t, s in names.items():
        weight = 1.0 if t in ARM_ALIGN else leg_align if t in LEG_ALIGN else 0.0
        tb = tgt.data.bones[t]
        tdir = ((tw @ tb.tail_local) - (tw @ tb.head_local)).normalized()
        sb = src.data.bones[s]
        sdir = mv(((sw @ sb.tail_local) - (sw @ sb.head_local)).normalized())
        align[t] = Quaternion().slerp(tdir.rotation_difference(sdir), weight)
    src_hip_rest = mv(sw @ src.data.bones[names['Hips']].head_local)
    tgt_hip_rest = (tw @ tgt.data.bones['Hips'].head_local)
    hip_scale = tgt_hip_rest.z / max(src_hip_rest.z, 1e-6)

    new_action = bpy.data.actions.new(name)
    new_action.use_fake_user = True
    tgt.animation_data_create()
    tgt.animation_data.action = new_action
    for pb in tgt.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    bone_order = order(tgt)
    inv_tw = tw.inverted()
    prev = {}
    hip_samples = []
    frames = list(range(start, end + 1))
    for f in frames:
        scene.frame_set(f)
        world = {}
        for t in bone_order:
            b = tgt.data.bones[t]
            if t in names:
                delta = mq(rot(sw @ src.pose.bones[names[t]].matrix)) @ src_rest[t].inverted()
                world[t] = delta @ align[t] @ tgt_rest[t]
        hip_now = mv(sw @ src.pose.bones[names['Hips']].head)
        hip_samples.append(tgt_hip_rest + (hip_now - src_hip_rest) * hip_scale)
        pose = {}
        for t in bone_order:
            b = tgt.data.bones[t]
            pb = tgt.pose.bones[t]
            rest_arm = b.matrix_local
            if b.parent is None:
                # a mount rig's root (horse_root) is not Mixamo's: it keeps its rest;
                # only a root Hips follows the source hips
                head = inv_tw @ hip_samples[-1] if t == 'Hips' else rest_arm.translation
            else:
                parent_pose = pose[b.parent.name]
                head = (parent_pose @ (b.parent.matrix_local.inverted() @ rest_arm)).translation
            if t in world:
                r_arm = (inv_tw.to_quaternion() @ world[t]).to_matrix().to_4x4()
            elif b.parent is None:
                r_arm = rest_arm.to_3x3().normalized().to_4x4()
            else:
                parent_pose = pose[b.parent.name]
                r_arm = (parent_pose @ (b.parent.matrix_local.inverted() @ rest_arm)).to_3x3().normalized().to_4x4()
            m = Matrix.Translation(head) @ r_arm
            pose[t] = m
            if b.parent is None:
                basis = rest_arm.inverted() @ m
            else:
                basis = (b.parent.matrix_local.inverted() @ rest_arm).inverted() @ pose[b.parent.name].inverted() @ m
            q = basis.to_quaternion()
            if t in prev:
                q.make_compatible(prev[t])
            prev[t] = q
            pb.rotation_quaternion = q
            if b.parent is None:
                pb.location = basis.translation
            pb.keyframe_insert('rotation_quaternion', frame=f - start)
            if b.parent is None:
                pb.keyframe_insert('location', frame=f - start)
    if in_place and tgt.data.bones['Hips'].parent is None:
        hips = tgt.pose.bones['Hips']
        first = frames[0]
        for f in frames:
            scene.frame_set(f)
        # Remove linear horizontal drift from the root so the loop closes in place.
        drift = hip_samples[-1] - hip_samples[0]
        for i, f in enumerate(frames):
            t = i / max(len(frames) - 1, 1)
            corrected = hip_samples[i] - Vector((drift.x * t, drift.y * t, 0))
            corrected.x = corrected.x - hip_samples[0].x + tgt_hip_rest.x
            corrected.y = corrected.y - hip_samples[0].y + tgt_hip_rest.y
            basis = tgt.data.bones['Hips'].matrix_local.inverted() @ (Matrix.Translation(inv_tw @ corrected))
            hips.location = basis.translation
            hips.keyframe_insert('location', frame=f - first)
    tgt.animation_data.action = None
    return {'name': name, 'frames': len(frames), 'seconds': round((len(frames) - 1) / scene.render.fps, 3),
            'inPlace': in_place, 'hipScale': round(hip_scale, 4)}


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(Path(args.target).resolve()))
    for o in list(bpy.context.scene.objects):
        if o.type == 'MESH' and not any(m.type == 'ARMATURE' for m in o.modifiers):
            bpy.data.objects.remove(o)  # glTF importer bone-shape helpers
    tgt = armature(bpy.context.scene.objects)
    existing = [a for a in bpy.data.actions]
    if not args.keep_existing:
        for a in existing:
            bpy.data.actions.remove(a)
    if tgt.animation_data:
        tgt.animation_data.action = None
    for pb in tgt.pose.bones:
        pb.matrix_basis.identity()
    clips = []
    for spec in args.clip:
        name, path = spec.split('=', 1)
        src, action, new = import_fbx(Path(path).resolve())
        info = retarget(tgt, src, action, name, args.leg_align, name in args.loop, name in args.mirror)
        info['mirrored'] = name in args.mirror
        info['source'] = Path(path).name
        clips.append(info)
        for o in new:
            bpy.data.objects.remove(o)
        bpy.data.actions.remove(action)
        print('RETARGETED', json.dumps(info))
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for pb in tgt.pose.bones:
        pb.matrix_basis.identity()
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True,
                              export_animation_mode='ACTIONS', export_anim_slide_to_zero=True,
                              export_force_sampling=True, export_frame_range=False)
    out.with_suffix('.json').write_text(json.dumps({'target': args.target, 'legAlign': args.leg_align,
                                                    'clips': clips}, indent=2) + '\n')


if __name__ == '__main__':
    main()
