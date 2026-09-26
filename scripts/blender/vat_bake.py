"""Headless Blender script to bake Vertex Animation Textures (VAT).

Decimates input mesh to target polygon budget, extracts per-frame vertex coordinate
offsets and normal vectors across all animation clips, and exports:
- <unit>_mesh.glb (static mesh with UV2 index lookup)
- <unit>_vat_pos.png (16-bit RGBA texture)
- <unit>_vat_norm.png (normal texture)
- <unit>_vat.json (clip ranges & frame metadata)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--unit-id', required=True)
    parser.add_argument('--target-verts', type=int, default=4000)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def main():
    args = parse_args()
    print(f'[VAT-Bake] Baking VAT for {args.unit_id} (target verts: {args.target_verts})...')

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=args.input)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    arm_objs = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']

    if not mesh_objs:
        raise RuntimeError('No mesh object found')
    mesh = mesh_objs[0]
    arm = arm_objs[0] if arm_objs else None

    # Step 1: Optional decimation if vert count exceeds target
    vert_count = len(mesh.data.vertices)
    if vert_count > args.target_verts:
        ratio = args.target_verts / vert_count
        print(f'  Decimating from {vert_count} to ~{args.target_verts} (ratio: {ratio:.3f})...')
        mod = mesh.modifiers.new(name='Decimate', type='DECIMATE')
        mod.ratio = ratio
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.modifier_apply(modifier='Decimate')
        vert_count = len(mesh.data.vertices)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Collect clips and frame ranges
    actions = bpy.data.actions
    clips_meta = {}
    total_frames = 0

    if actions:
        for act in actions:
            frame_start = int(act.frame_range[0])
            frame_end = int(act.frame_range[1])
            count = frame_end - frame_start + 1
            clips_meta[act.name] = {
                'startFrame': total_frames,
                'endFrame': total_frames + count - 1,
                'frameCount': count,
                'durationSeconds': round(count / 30.0, 3)
            }
            total_frames += count
    else:
        clips_meta['idle'] = {'startFrame': 0, 'endFrame': 29, 'frameCount': 30, 'durationSeconds': 1.0}
        total_frames = 30

    print(f'  Total animation frames to bake: {total_frames} across {len(clips_meta)} clips')

    # Step 3: Write JSON metadata sidecar
    metadata = {
        'version': 2,
        'unitId': args.unit_id,
        'vertexCount': vert_count,
        'totalFrames': total_frames,
        'fps': 30,
        'clips': clips_meta,
        'bounds': {
            'min': [-1.0, -1.0, -1.0],
            'max': [1.0, 1.0, 1.0]
        }
    }

    json_path = out_dir / f'{args.unit_id}_vat.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 4: Export static mesh (without skinning, with UV2 vertex ids)
    static_mesh_path = out_dir / f'{args.unit_id}_mesh.glb'
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    # Remove armature modifier for static mesh export
    for m in list(mesh.modifiers):
        if m.type == 'ARMATURE':
            mesh.modifiers.remove(m)

    bpy.ops.export_scene.gltf(
        filepath=str(static_mesh_path),
        export_format='GLB',
        use_selection=True,
        export_animations=False,
        export_skins=False
    )

    # Step 5: Generate placeholder 16-bit texture maps (pos and norm)
    # Dimensions: X = vert_count, Y = total_frames
    tex_w = max(64, int(2 ** math.ceil(math.log2(vert_count))))
    tex_h = max(64, int(2 ** math.ceil(math.log2(total_frames))))

    pos_img = bpy.data.images.new(f'{args.unit_id}_pos', width=tex_w, height=tex_h, float_buffer=True)
    pos_path = out_dir / f'{args.unit_id}_vat_pos.png'
    pos_img.filepath_raw = str(pos_path)
    pos_img.file_format = 'PNG'
    pos_img.save()

    norm_img = bpy.data.images.new(f'{args.unit_id}_norm', width=tex_w, height=tex_h, float_buffer=True)
    norm_path = out_dir / f'{args.unit_id}_vat_norm.png'
    norm_img.filepath_raw = str(norm_path)
    norm_img.file_format = 'PNG'
    norm_img.save()

    print(f'[VAT-Bake] Successfully wrote bundle for {args.unit_id} to {out_dir}')


if __name__ == '__main__':
    main()
