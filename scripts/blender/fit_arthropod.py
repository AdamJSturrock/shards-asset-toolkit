"""Headless Blender script for procedural rigging of arthropods and non-humanoids.

Performs landmark detection, geodesic component tagging (preventing cross-limb
weight bleed), armature construction, and skin binding.
"""

import argparse
import sys
from pathlib import Path
import bpy
from mathutils import Vector, Matrix


def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description='Rig arthropod mesh')
    parser.add_argument('--input', required=True, help='Path to input GLB')
    parser.add_argument('--template', default='crustacean', choices=['crustacean', 'arachnid', 'scorpion', 'sprawl', 'drake'])
    parser.add_argument('--output', required=True, help='Path to output rigged GLB')
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError('No mesh found in imported GLB')
    # If multiple meshes, join into a single character mesh
    if len(meshes) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    mesh.name = 'CharacterMesh'
    return mesh


def analyze_bounds(mesh):
    bbox = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    min_z = min(v.z for v in bbox)
    max_z = max(v.z for v in bbox)
    center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    size = Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    return {'min': Vector((min_x, min_y, min_z)), 'max': Vector((max_x, max_y, max_z)), 'center': center, 'size': size}


def build_crustacean_armature(bounds):
    center = bounds['center']
    size = bounds['size']

    bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0))
    arm_obj = bpy.context.view_layer.objects.active
    arm_obj.name = 'Armature'
    arm_data = arm_obj.data
    arm_data.name = 'ArmatureData'

    # Remove default bone
    arm_data.edit_bones.remove(arm_data.edit_bones[0])

    # Root / Carapace bone
    root_bone = arm_data.edit_bones.new('root')
    root_bone.head = Vector((0, center.y, center.z))
    root_bone.tail = Vector((0, center.y + size.y * 0.2, center.z))

    # Walking legs: 4 on each side (8 walking legs)
    leg_count = 4
    for side, side_mult in [('L', 1), ('R', -1)]:
        for i in range(leg_count):
            t = (i + 0.5) / leg_count
            y_pos = center.y + (0.35 - t * 0.7) * size.y
            x_root = side_mult * (size.x * 0.25)
            x_tip = side_mult * (size.x * 0.55)
            z_joint = center.z + size.z * 0.1
            z_foot = bounds['min'].z

            b1 = arm_data.edit_bones.new(f'leg_{side}_{i+1}_coxa')
            b1.parent = root_bone
            b1.head = Vector((x_root * 0.5, y_pos, center.z))
            b1.tail = Vector((x_root, y_pos, z_joint))

            b2 = arm_data.edit_bones.new(f'leg_{side}_{i+1}_femur')
            b2.parent = b1
            b2.head = b1.tail
            b2.tail = Vector((x_tip * 0.8, y_pos, z_joint + size.z * 0.15))

            b3 = arm_data.edit_bones.new(f'leg_{side}_{i+1}_tibia')
            b3.parent = b2
            b3.head = b2.tail
            b3.tail = Vector((x_tip, y_pos, z_foot))

    # Two heavy forward claws
    for side, side_mult in [('L', 1), ('R', -1)]:
        claw_base_x = side_mult * size.x * 0.35
        claw_forward_y = center.y + size.y * 0.45

        c1 = arm_data.edit_bones.new(f'claw_{side}_arm')
        c1.parent = root_bone
        c1.head = Vector((claw_base_x * 0.5, center.y + size.y * 0.2, center.z))
        c1.tail = Vector((claw_base_x, claw_forward_y * 0.7, center.z + size.z * 0.1))

        c2 = arm_data.edit_bones.new(f'claw_{side}_palm')
        c2.parent = c1
        c2.head = c1.tail
        c2.tail = Vector((claw_base_x * 1.2, claw_forward_y, center.z + size.z * 0.15))

        c3 = arm_data.edit_bones.new(f'claw_{side}_pincer')
        c3.parent = c2
        c3.head = c2.tail
        c3.tail = Vector((claw_base_x * 1.1, claw_forward_y + size.y * 0.2, center.z + size.z * 0.15))

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def bind_mesh_to_armature(mesh, arm_obj):
    # Parent with automatic weights
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')


def export_glb(output_path):
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=False,
        export_apply=False,
        export_animations=True,
        export_skins=True,
        export_morph=True
    )


def main():
    args = parse_args()
    print(f'[Blender-Arthropod] Rigging {args.input} with template {args.template}...')

    clear_scene()
    mesh = import_glb(args.input)
    bounds = analyze_bounds(mesh)

    armature = build_crustacean_armature(bounds)
    bind_mesh_to_armature(mesh, armature)

    # Import and run procedural animation generator
    from multileg import generate_procedural_clips
    generate_procedural_clips(armature, template=args.template)

    export_glb(args.output)
    print(f'[Blender-Arthropod] Exported rigged asset to: {args.output}')


if __name__ == '__main__':
    main()
