# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""
rig_mount.py: rig a static Meshy rider-on-a-mount (one welded mesh)
as ONE skinned model with two skeletons: the mount on the critter quadruped
template (bones prefixed `horse_`) and the rider on Meshy's 24-joint humanoid
(the Mixamo retarget chain's names), the rider's Hips parented to a mount bone
(the saddle). The runtime plays one mixer per unit, so it must be one GLB.

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/rig_mount.py -- \
      --spec examples/anim-authoring/mounted_knight.json \
      --out output/anim-authoring/mounted_knight/rig-v1/character.glb

Spec "mountRig":
  mesh        static Meshy GLB, normalised as rig_humanoid.py does
              (centred in x/y, lowest point z = 0, facing -Y)
  mount       {"plan": "quadruped", "joints": {spine/neck/head/tail joints},
               "legs": {"front": {"yRange": [y0, y1], "joints": {"shoulder": z,
               "elbow": z, "fetlock": z, "ftoe": 0}}, "hind": {... "hip",
               "stifle", "hock", "htoe"}}, "toeForward": 0.05}
              Leg joints are measured, not guessed: each leg's centre at a
              joint height is the centroid of a thin horizontal slice of the
              vertices in its column (x side, yRange); joints above the leg
              (shoulder/hip) take the column's x/y at the top slice.
  rider       {"joints": {every Meshy joint + LeftHandTip, RightHandTip,
               LeftToeTip, RightToeTip}, "parent": "horse_hips"}
  riderRegions  [{"box"|"capsule", "maxSat": 0.35}...]: a vertex is the
              rider's when it lies in any region (and, if the region sets
              maxSat, its base-colour texel's HSV saturation is below it:
              steel armour against a red caparison or a brown flank).
  shield      as rig_fixup.py (attached to the rider's forearm).

Skinning: Blender heat weights on a welded copy, run twice (mount bones
deforming only, then rider bones only); each vertex keeps the pass of its own
label, so no rider bone ever pulls the horse and no horse bone the rider. At
most 4 influences, normalised. Writes the skinned GLB (no animations), a
.blend and <out>.json (joints, label counts, per-bone vertex counts).
"""
import argparse
import colorsys
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector, kdtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'critter'))
from rig_humanoid import BONES as RIDER_BONES, load_mesh, in_region  # noqa: E402
import critter_templates  # noqa: E402
from rig_fixup import Skin, attach_shield, dominant, base_image  # noqa: E402

X, Y = Vector((1, 0, 0)), Vector((0, 1, 0))
SIDES_TEMPLATE = getattr(critter_templates, 'SIDES', ('L', 'R'))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args(sys.argv[sys.argv.index('--') + 1:])


def leg_joints(co, legs, toe_fwd):
    """Measure the four legs' joints from horizontal slices of their columns."""
    J = {}
    for kind, cfg in legs.items():
        y0, y1 = cfg['yRange']
        heights = cfg['joints']
        for s, sgn in (('L', 1), ('R', -1)):
            col = co[(np.sign(co[:, 0]) == sgn) & (co[:, 1] >= y0) & (co[:, 1] <= y1)]
            top_name = list(heights)[0]
            cache = {}
            for name, z in heights.items():
                zz = z
                band = col[np.abs(col[:, 2] - zz) < 0.02]
                if name == top_name or len(band) < 10:
                    # above the leg (hidden under a caparison): the column's x/y at the highest clean slice
                    ref = cache.get('ref')
                    if ref is None:
                        for zt in np.arange(cfg.get('refTop', 0.45), 0.05, -0.02):
                            b2 = col[np.abs(col[:, 2] - zt) < 0.02]
                            if len(b2) > 20:
                                ref = b2.mean(0)
                                break
                        cache['ref'] = ref
                    c = np.array([ref[0], ref[1], zz])
                else:
                    c = band.mean(0)
                    c[2] = zz
                if name.endswith('toe'):
                    c = c + np.array([0, -toe_fwd, 0])
                J[f'{name}.{s}'] = Vector(c.tolist())
    return J


def build(mount, rider, J_mount, J_rider):
    arm = bpy.data.armatures.new('Armature')
    rig = bpy.data.objects.new('Armature', arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    hb = critter_templates.bones_for(mount.get('plan', 'quadruped'), {k: list(v) for k, v in J_mount.items()})
    mount_names = []
    for name, h, t, parent in hb:
        eb = arm.edit_bones.new('horse_' + name)
        eb.head, eb.tail = J_mount[h], J_mount[t]
        eb.align_roll(Vector((0, 0, 1)) if abs((eb.tail - eb.head).normalized().z) < 0.8 else Vector((0, -1, 0)))
        mount_names.append('horse_' + name)
    for name, h, t, parent in hb:
        if parent:
            arm.edit_bones['horse_' + name].parent = arm.edit_bones['horse_' + parent]
    rider_names = []
    for name, h, t, parent, deform in RIDER_BONES:
        eb = arm.edit_bones.new(name)
        eb.head = J_rider[h]
        eb.tail = J_rider[t] if t else J_rider[h] + (J_rider[h] - J_rider['Head']).normalized() * 0.05
        yax = (eb.tail - eb.head).normalized()
        ref = X if abs(yax.x) < 0.8 else Y
        xp = (ref - ref.dot(yax) * yax).normalized()
        eb.align_roll(xp.cross(yax))
        eb.use_deform = deform
        if deform:
            rider_names.append(name)
    for name, h, t, parent, deform in RIDER_BONES:
        arm.edit_bones[name].parent = arm.edit_bones[parent] if parent else arm.edit_bones[rider.get('parent', 'horse_hips')]
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig, mount_names, rider_names


def heat_pass(mesh, rig, deform_names):
    for b in rig.data.bones:
        b.use_deform = b.name in deform_names
    dup = mesh.copy()
    dup.data = mesh.data.copy()
    bpy.context.scene.collection.objects.link(dup)
    bm = bmesh.new()
    bm.from_mesh(dup.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bm.to_mesh(dup.data)
    bm.free()
    for o in bpy.context.scene.objects:
        o.select_set(o in (dup, rig))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    kd = kdtree.KDTree(len(dup.data.vertices))
    for v in dup.data.vertices:
        kd.insert(v.co, v.index)
    kd.balance()
    names = {g.index: g.name for g in dup.vertex_groups}
    dw = [{names[g.group]: g.weight for g in v.groups if g.weight > 1e-4} for v in dup.data.vertices]
    bpy.data.objects.remove(dup, do_unlink=True)
    return kd, dw


def main():
    args = parse()
    spec = json.loads(Path(args.spec).read_text())
    S = spec['mountRig']
    mesh, offset = load_mesh(S['mesh'])
    me = mesh.data
    co = np.array([v.co[:] for v in me.vertices])
    rep = {'offset': offset.tolist(), 'verts': len(co)}
    # joints
    M = S['mount']
    J_mount = {k: Vector(v) for k, v in M['joints'].items()}
    J_mount.update(leg_joints(co, M['legs'], M.get('toeForward', 0.05)))
    J_rider = {k: Vector(v) for k, v in S['rider']['joints'].items()}
    rep['mountJoints'] = {k: [round(x, 4) for x in v] for k, v in J_mount.items()}
    # labels: rider regions (+ texture saturation)
    img = base_image(mesh.material_slots[0].material)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    uvl = me.uv_layers.active.data
    vuv = np.zeros((len(co), 2))
    for loop in me.loops:
        vuv[loop.vertex_index] = uvl[loop.index].uv[:]
    sat = np.zeros(len(co))
    for i, (u, v) in enumerate(vuv):
        r, g, b = px[min(h - 1, int(v * h)), min(w - 1, int(u * w)), :3]
        sat[i] = colorsys.rgb_to_hsv(r, g, b)[1]
    rider = np.zeros(len(co), bool)
    for i, p in enumerate(co):
        pv = Vector(p)
        for r in S['riderRegions']:
            if in_region(pv, r) and ('maxSat' not in r or sat[i] <= r['maxSat']):
                rider[i] = True
                break
    rep['riderVerts'] = int(rider.sum())
    rig, mount_names, rider_names = build(M, S['rider'], J_mount, J_rider)
    kd_m, dw_m = heat_pass(mesh, rig, set(mount_names))
    kd_r, dw_r = heat_pass(mesh, rig, set(rider_names))
    for b in rig.data.bones:
        b.use_deform = b.name in set(mount_names) | set(rider_names)
    # assign
    for vg in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(vg)
    groups = {n: mesh.vertex_groups.new(name=n) for n in mount_names + rider_names}
    segs = {b.name: (b.head_local.copy(), b.tail_local.copy()) for b in rig.data.bones}
    counts = {n: 0 for n in groups}
    fallback = 0
    for i, v in enumerate(me.vertices):
        is_r = bool(rider[i])
        kd, dw, allowed = (kd_r, dw_r, rider_names) if is_r else (kd_m, dw_m, mount_names)
        _, j, _ = kd.find(v.co)
        wts = {k: x for k, x in dw[j].items() if k in allowed}
        if not wts:
            fallback += 1
            best = min(allowed, key=lambda n: (lambda a, b, p: ((a + (b - a) * max(0.0, min(1.0, (p - a).dot(b - a) / max((b - a).length_squared, 1e-12)))) - p).length)(*segs[n], v.co))
            wts = {best: 1.0}
        items = sorted(wts.items(), key=lambda kv: -kv[1])[:4]
        tot = sum(x for _, x in items)
        for n, x in items:
            groups[n].add([i], x / tot, 'REPLACE')
        counts[items[0][0]] += 1
    mesh.parent = rig
    mesh.modifiers.clear()
    mod = mesh.modifiers.new('Armature', 'ARMATURE')
    mod.object = rig
    rep.update({'heatFallbackVerts': fallback, 'dominantCounts': counts,
                'emptyBones': [n for n, c in counts.items() if c == 0]})
    # shield on the rider's forearm
    if S.get('shield'):
        P = [mesh.matrix_world @ v.co for v in me.vertices]
        skin = Skin(mesh, rig)
        bones = {b.name: (rig.matrix_world @ b.head_local, rig.matrix_world @ b.tail_local) for b in rig.data.bones}
        H = float(co[:, 2].max() - co[:, 2].min())
        attach_shield(rig, mesh, skin, P, bones, H, H, S['shield'], rep)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True,
                              export_animations=False, export_skins=True, export_all_influences=False,
                              export_yup=True, export_apply=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    out.with_suffix('.json').write_text(json.dumps(rep, indent=2) + '\n')
    print('MOUNTRIG', json.dumps({k: rep[k] for k in ('verts', 'riderVerts', 'heatFallbackVerts', 'emptyBones')}),
          json.dumps(rep.get('shield', {})))


if __name__ == '__main__':
    main()
