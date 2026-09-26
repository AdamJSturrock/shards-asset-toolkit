# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Audit and repair head skin weights on Meshy humanoid GLBs.

Meshy's auto-rig often leaks shoulder influence into large heads and ears:
A goblin unit had ~3,800 head vertices 10-20% bound to Left/RightShoulder, so
any shoulder motion sheared the face and bent the ears. The repair:

1. Head core (Head weight >= --core, default 0.5) becomes rigid: Head = 1.
2. Transition vertices (0 < Head < core) lose shoulder/arm/hips influence;
   the remainder is renormalised over Head, neck and the spine chain, so the
   neck still bends smoothly.

Audit only (no output file):

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/repair_head_weights.py -- \
      --input input/goblin_worker/meshy_rigged.glb

Repair (writes a new GLB; never overwrites the input):

    ... -- --input <character.glb> --out output/anim-authoring/goblin_worker/rig-v1/character.glb

The report (printed, and <out>.json with --out) lists head vertex counts by
Head weight, the leaking bones, and a pose test: the head is turned 30 degrees
and the shoulders raised 25 degrees; vertices fully bound to Head must then move
rigidly with it (max deviation reported in model units). The measured set
is fixed before repair, so the leak shows as a large 'before' deviation. Inspect textured
close-ups of a head-turning clip before promoting the result.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Euler

LEAK_BONES = ('LeftShoulder', 'RightShoulder', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
              'LeftHand', 'RightHand', 'Hips', 'LeftUpLeg', 'RightUpLeg')
KEEP_BONES = ('Head', 'neck', 'Spine', 'Spine01', 'Spine02')


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(Path(path).resolve()))
    for o in list(bpy.context.scene.objects):
        if o.type == 'MESH' and not any(m.type == 'ARMATURE' for m in o.modifiers):
            bpy.data.objects.remove(o)  # importer bone-shape helpers
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    return rig, meshes


def weights(mesh, v):
    return {mesh.vertex_groups[g.group].name: g.weight for g in v.groups}


def audit(meshes, core):
    hist, leak, core_n, partial_n = Counter(), Counter(), 0, 0
    for mesh in meshes:
        for v in mesh.data.vertices:
            ws = weights(mesh, v)
            h = ws.get('Head', 0.0)
            if h <= 0.02:
                continue
            hist[round(h, 1)] += 1
            if h >= core:
                core_n += 1
                for k, w in ws.items():
                    if k in LEAK_BONES and w > 0.01:
                        leak[k] += 1
            elif h < 0.98:
                partial_n += 1
    return {'headVerts': sum(hist.values()), 'coreVerts': core_n, 'transitionVerts': partial_n,
            'coreVertsWithLeak': dict(leak.most_common()), 'headWeightHistogram': sorted(hist.items())}


def repair(meshes, core):
    changed = 0
    for mesh in meshes:
        groups = {g.name: g for g in mesh.vertex_groups}
        head = groups['Head']
        for v in mesh.data.vertices:
            ws = weights(mesh, v)
            h = ws.get('Head', 0.0)
            if h <= 0.0:
                continue
            if h >= core:
                new = {'Head': 1.0}
            else:
                kept = {k: w for k, w in ws.items() if k in KEEP_BONES}
                total = sum(kept.values())
                if total <= 0:
                    continue
                new = {k: w / total for k, w in kept.items()}
            if all(abs(ws.get(k, 0) - new.get(k, 0)) < 1e-4 for k in set(ws) | set(new)):
                continue
            for name in ws:
                if name not in new:
                    groups[name].remove([v.index])
            for name, w in new.items():
                groups.get(name, head).add([v.index], w, 'REPLACE')
            changed += 1
    return changed


def head_region(meshes, core):
    """Vertex indices whose ORIGINAL Head weight is >= core: the set the pose test measures."""
    return {m.name: {v.index for v in m.data.vertices if weights(m, v).get('Head', 0) >= core} for m in meshes}


def pose_test(rig, meshes, region):
    """Turn the head, raise the shoulders; the head region must follow Head rigidly.

    The region is fixed before repair, so the same vertices are measured before
    and after: a leaking rig fails this test, a repaired one passes."""
    dg = bpy.context.evaluated_depsgraph_get()
    rest = {m.name: [m.matrix_world @ v.co for v in m.data.vertices] for m in meshes}
    head_rest = (rig.matrix_world @ rig.pose.bones['Head'].matrix).copy()
    pb = rig.pose.bones
    for p in pb:
        p.rotation_mode = 'XYZ'
    pb['Head'].rotation_euler = Euler((0, 0, math.radians(30)))
    for side in ('LeftShoulder', 'RightShoulder'):
        if side in pb:
            pb[side].rotation_euler = Euler((0, 0, math.radians(25 if side.startswith('Left') else -25)))
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    head_now = rig.matrix_world @ pb['Head'].matrix
    delta = head_now @ head_rest.inverted()
    worst = 0.0
    for m in meshes:
        ev = m.evaluated_get(dg)
        em = ev.to_mesh()
        for v, ve in zip(m.data.vertices, em.vertices):
            if v.index in region[m.name]:
                expected = delta @ rest[m.name][v.index]
                worst = max(worst, ((m.matrix_world @ ve.co) - expected).length)
        ev.to_mesh_clear()
    for p in pb:
        p.rotation_euler = Euler((0, 0, 0))
    bpy.context.view_layer.update()
    height = max((m.matrix_world @ v.co).z for m in meshes for v in m.data.vertices)
    return {'coreMaxDeviation': round(worst, 5), 'coreMaxDeviationPctHeight': round(100 * worst / height, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out')
    parser.add_argument('--core', type=float, default=0.5)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    rig, meshes = load(args.input)
    if rig.animation_data:
        rig.animation_data.action = None
    region = head_region(meshes, args.core)
    report = {'input': args.input, 'before': audit(meshes, args.core), 'posedBefore': pose_test(rig, meshes, region)}
    if args.out:
        out = Path(args.out).resolve()
        if out.exists() or out == Path(args.input).resolve():
            raise RuntimeError('Refusing to overwrite; choose a new output path')
        report['changedVerts'] = repair(meshes, args.core)
        report['after'] = audit(meshes, args.core)
        report['posedAfter'] = pose_test(rig, meshes, region)
        out.parent.mkdir(parents=True, exist_ok=True)
        for a in list(bpy.data.actions):
            bpy.data.actions.remove(a)
        bpy.ops.object.select_all(action='DESELECT')
        for o in [rig, *meshes]:
            o.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True, export_animations=False)
        out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('HEAD_WEIGHTS', json.dumps(report))


if __name__ == '__main__':
    main()
