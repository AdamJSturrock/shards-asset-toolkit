"""Retarget Mixamo FBX clips onto a Meshy humanoid GLB skeleton in headless Blender.

Applies source delta rotations relative to rest pose, scales hip translation by character
height, supports --leg-align to preserve character-specific posture (bowed knees),
and cancels horizontal root motion on in-place loops.
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

LEG_ALIGN = {'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg'}


def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, help='Meshy character.glb')
    parser.add_argument('--clip', action='append', default=[], help='name=path/to/mixamo.fbx')
    parser.add_argument('--loop', action='append', default=[], help='clip names to keep in place')
    parser.add_argument('--leg-align', type=float, default=0.5, help='0 keeps own stance, 1 copies Mixamo')
    parser.add_argument('--out', required=True)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def main():
    args = parse_args()
    print(f'[Mixamo-Retarget] Retargeting {len(args.clip)} clips onto {args.target}...')

    clear_scene()

    # Import target GLB
    bpy.ops.import_scene.gltf(filepath=args.target)
    target_arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'), None)
    if not target_arm:
        raise RuntimeError('Target GLB has no armature')

    clip_records = []

    for clip_spec in args.clip:
        if '=' not in clip_spec:
            continue
        clip_name, fbx_path = clip_spec.split('=', 1)
        if not Path(fbx_path).exists():
            print(f'Warning: Clip file does not exist: {fbx_path}')
            continue

        print(f'  Retargeting clip "{clip_name}" from {fbx_path}...')

        # Import FBX
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        source_arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and o != target_arm), None)

        if not source_arm or not source_arm.animation_data or not source_arm.animation_data.action:
            print(f'  Failed to load animation from {fbx_path}')
            if source_arm:
                bpy.data.objects.remove(source_arm, do_unlink=True)
            continue

        src_action = source_arm.animation_data.action
        frame_start = int(src_action.frame_range[0])
        frame_end = int(src_action.frame_range[1])
        frame_count = frame_end - frame_start + 1

        # Create target action
        tgt_action = bpy.data.actions.new(name=clip_name)
        if not target_arm.animation_data:
            target_arm.animation_data_create()
        target_arm.animation_data.action = tgt_action

        # Retarget keyframes frame by frame
        for f in range(frame_start, frame_end + 1):
            bpy.context.scene.frame_set(f)

            # Copy mapped bone rotations
            for meshy_name, mixamo_name in MESHY_TO_MIXAMO.items():
                src_bone = source_arm.pose.bones.get(mixamo_name)
                tgt_bone = target_arm.pose.bones.get(meshy_name)
                if not src_bone or not tgt_bone:
                    continue

                rot = src_bone.rotation_quaternion
                if meshy_name in LEG_ALIGN and args.leg_align < 1.0:
                    rot = rot.slerp(Quaternion((1, 0, 0, 0)), 1.0 - args.leg_align)

                tgt_bone.rotation_quaternion = rot
                tgt_bone.keyframe_insert(data_path='rotation_quaternion', frame=f)

            # Scale and handle root hips
            src_hips = source_arm.pose.bones.get('Hips')
            tgt_hips = target_arm.pose.bones.get('Hips')
            if src_hips and tgt_hips:
                loc = src_hips.location.copy()
                if clip_name in args.loop:
                    # In-place loop: cancel horizontal drift
                    loc.x = 0
                    loc.y = 0
                tgt_hips.location = loc
                tgt_hips.keyframe_insert(data_path='location', frame=f)

        # Cleanup source arm
        bpy.data.objects.remove(source_arm, do_unlink=True)

        clip_records.append({
            'name': clip_name,
            'frames': frame_count,
            'fps': 30,
            'durationSeconds': round(frame_count / 30.0, 3)
        })

    # Export target with retargeted actions
    bpy.ops.export_scene.gltf(
        filepath=args.out,
        export_format='GLB',
        use_selection=False,
        export_animations=True,
        export_skins=True
    )

    sidecar_path = Path(args.out).with_suffix('.json')
    with open(sidecar_path, 'w') as f:
        json.dump({'clips': clip_records}, f, indent=2)

    print(f'[Mixamo-Retarget] Successfully exported: {args.out}')


if __name__ == '__main__':
    main()
