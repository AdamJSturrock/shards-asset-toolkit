"""Procedural gait and animation authoring for multi-legged creatures in Blender.

Generates idle, walk, attack, and death clips with exact loop seam closure.
"""

import math
import bpy
from mathutils import Euler, Vector


def create_action(armature, action_name):
    if not armature.animation_data:
        armature.animation_data_create()
    action = bpy.data.actions.new(name=action_name)
    armature.animation_data.action = action
    return action


def generate_idle(armature, fps=30, duration_frames=40):
    action = create_action(armature, 'idle')
    root = armature.pose.bones.get('root')

    for f in range(duration_frames + 1):
        phase = (f / duration_frames) * 2 * math.pi
        z_offset = math.sin(phase) * 0.03
        pitch = math.sin(phase) * 0.02

        if root:
            root.location = Vector((0, 0, z_offset))
            root.keyframe_insert(data_path='location', frame=f)
            root.rotation_euler = Euler((pitch, 0, 0), 'XYZ')
            root.keyframe_insert(data_path='rotation_euler', frame=f)

    return action


def generate_walk(armature, template='crustacean', fps=30, duration_frames=30):
    action = create_action(armature, 'walk')
    root = armature.pose.bones.get('root')

    # Lateral scuttle for crabs; forward tetrapod for spiders
    is_scuttle = (template == 'crustacean')

    for f in range(duration_frames + 1):
        phase = (f / duration_frames) * 2 * math.pi

        # Body bob
        if root:
            bob = math.sin(phase * 2) * 0.04
            roll = math.sin(phase) * 0.03 if is_scuttle else 0
            root.location = Vector((0, 0, bob))
            root.keyframe_insert(data_path='location', frame=f)
            root.rotation_euler = Euler((0, roll, 0), 'XYZ')
            root.keyframe_insert(data_path='rotation_euler', frame=f)

        # Animate leg pairs
        for side, side_mult in [('L', 1), ('R', -1)]:
            for i in range(4):
                leg_name = f'leg_{side}_{i+1}_femur'
                femur = armature.pose.bones.get(leg_name)
                if not femur:
                    continue

                # Antiphase offsets between adjacent legs
                leg_phase = phase + (i * math.pi * 0.5) + (math.pi if side == 'R' else 0)
                lift = max(0.0, math.sin(leg_phase)) * 0.25
                swing = math.cos(leg_phase) * 0.2

                femur.rotation_euler = Euler((swing, 0, lift * side_mult), 'XYZ')
                femur.keyframe_insert(data_path='rotation_euler', frame=f)

    return action


def generate_attack(armature, fps=30, duration_frames=24):
    action = create_action(armature, 'attack')
    root = armature.pose.bones.get('root')
    claw_pincer = armature.pose.bones.get('claw_R_pincer')
    claw_arm = armature.pose.bones.get('claw_R_arm')

    for f in range(duration_frames + 1):
        t = f / duration_frames
        if t < 0.3:
            # Wind up
            lift = (t / 0.3) * 0.4
            pincer_open = (t / 0.3) * 0.5
        elif t < 0.45:
            # Strike & Snap
            snap_t = (t - 0.3) / 0.15
            lift = 0.4 - snap_t * 0.6
            pincer_open = 0.5 - snap_t * 0.6
        else:
            # Recover
            rec_t = (t - 0.45) / 0.55
            lift = -0.2 + rec_t * 0.2
            pincer_open = -0.1 + rec_t * 0.1

        if claw_arm:
            claw_arm.rotation_euler = Euler((lift, 0, 0), 'XYZ')
            claw_arm.keyframe_insert(data_path='rotation_euler', frame=f)

        if claw_pincer:
            claw_pincer.rotation_euler = Euler((0, 0, pincer_open), 'XYZ')
            claw_pincer.keyframe_insert(data_path='rotation_euler', frame=f)

    return action


def generate_death(armature, fps=30, duration_frames=30):
    action = create_action(armature, 'death')
    root = armature.pose.bones.get('root')

    for f in range(duration_frames + 1):
        t = min(1.0, f / (duration_frames * 0.8))
        # Ease out drop
        drop = -(t * t) * 0.3

        if root:
            root.location = Vector((0, 0, drop))
            root.keyframe_insert(data_path='location', frame=f)

        # Legs curl inwards
        for side in ['L', 'R']:
            for i in range(4):
                femur = armature.pose.bones.get(f'leg_{side}_{i+1}_femur')
                tibia = armature.pose.bones.get(f'leg_{side}_{i+1}_tibia')
                curl = t * 0.6
                if femur:
                    femur.rotation_euler = Euler((0, 0, curl if side == 'L' else -curl), 'XYZ')
                    femur.keyframe_insert(data_path='rotation_euler', frame=f)
                if tibia:
                    tibia.rotation_euler = Euler((curl, 0, 0), 'XYZ')
                    tibia.keyframe_insert(data_path='rotation_euler', frame=f)

    return action


def generate_procedural_clips(armature, template='crustacean'):
    print(f'[Blender-MultiLeg] Generating procedural animation clips for {template}...')
    generate_idle(armature)
    generate_walk(armature, template=template)
    generate_attack(armature)
    generate_death(armature)
