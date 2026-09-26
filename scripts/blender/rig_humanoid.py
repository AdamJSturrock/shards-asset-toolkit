# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Rig a static Meshy humanoid on the Meshy 24-joint skeleton, in Blender.

For characters Meshy's auto-rig refuses ("Pose estimation failed"): the
Shellback crab-folk, whose head is fused into the carapace and whose right arm
is a claw. The skeleton copies the names and hierarchy of Meshy's humanoid
auto-rig, so the rest of the humanoid pipeline runs unchanged:
repair_head_weights.py -> retarget_mixamo.py ->
polish_clips.py -> clip_review.py. Skip the head repair when
a region has a "web": the repair strips arm influence from every vertex with
some Head weight, which flattens the web's gradient back into a hard seam.

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/rig_humanoid.py -- \
      --spec examples/anim-authoring/shellback_cutthroat.json \
      --out output/anim-authoring/shellback_cutthroat/rig-v1/character.glb

Reads the spec's "rigBuild" block:
  mesh     static Meshy GLB (normalised here: centred in x/y, lowest point at
           z = 0, facing -Y as Meshy exports it; glTF +Z)
  joints   {name: [x, y, z]} in that normalised space. Every Meshy joint, plus
           the tips LeftHandTip, RightHandTip, LeftToeTip, RightToeTip.
  boneLimits  optional {bone: {"zMin": .., "xMax": .., ...}}: heat weight of that
           bone is dropped outside the range (heat lets a Head bone sitting in a
           carapace reach down the belly and into the shoulders); the rest renormalises.
  regions  optional overrides applied after heat weighting, in order:
           {"note": "...", "box": [[x0,y0,z0],[x1,y1,z1]] | "capsule": [[a],[b],r],
            "where": {"xMin":..,"xMax":..,"yMin":..,"yMax":..,"zMin":..,"zMax":..},
            "weights": {"Head": 1.0}}
           Vertices inside take exactly those weights: rigid parts such as a
           carapace, a claw or a weapon fused to a hand.
           "radialNormal": 0.3 (capsules only) keeps just the vertices whose
           normal points away from the capsule axis (dot >= value): the
           surface of a thin prop, not the chest behind it or a claw face
           pressed against it. Lets the radius be generous.
           "grow": {"zMin": .., "normalZMax": -0.3, "foldRings": 2, "excludeDominant": [bones]}
           (optional) extends a rigid region past its box along the mesh: first
           `foldRings` edge rings (a carapace rim folding under), then any
           connected vertex whose normal's z is below normalZMax (the shell's
           underside). Vertices under zMin, or whose heat-dominant bone is in
           excludeDominant (the arms), are never added.
           "web": {"distance": 0.1, "iterations": 400} (optional) blends the
           vertices outside the region within `distance` (geodesic, along the
           mesh) from the region's rigid weights to their own heat weights: a
           harmonic gradient like a shoulder or neck, so the faces Meshy fused
           between a rigid part and the body stretch smoothly instead of tearing.
           Vertices of other rigid regions are left alone.

Faces are never deleted. Meshy fuses touching parts into one shell, and an
earlier "detachFrom" option cut those webs out: that left open boundaries on a
single-sided surface, i.e. see-through holes along the Shellback carapace rim.
"detachFrom"/"detachPairs" now raise. The export asserts the welded
boundary-edge count is no higher than the source mesh's.

Skinning is the critter-authoring method: Blender heat weights on a WELDED copy
(the glTF import splits vertices along UV seams, which heat diffusion reads as
holes), copied back by nearest vertex; anything heat could not reach falls back
to its nearest bone; at most 4 influences, normalised.

Writes the skinned GLB (no animations) and <out>.json with the joint table,
per-bone vertex counts and skin checks. Look at a posed render before trusting it.
"""
import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector, kdtree

# (bone, head joint, tail joint, parent, deform) in Meshy's order and hierarchy.
BONES = [
    ('Hips', 'Hips', 'Spine02', None, True),
    ('LeftUpLeg', 'LeftUpLeg', 'LeftLeg', 'Hips', True),
    ('LeftLeg', 'LeftLeg', 'LeftFoot', 'LeftUpLeg', True),
    ('LeftFoot', 'LeftFoot', 'LeftToeBase', 'LeftLeg', True),
    ('LeftToeBase', 'LeftToeBase', 'LeftToeTip', 'LeftFoot', True),
    ('RightUpLeg', 'RightUpLeg', 'RightLeg', 'Hips', True),
    ('RightLeg', 'RightLeg', 'RightFoot', 'RightUpLeg', True),
    ('RightFoot', 'RightFoot', 'RightToeBase', 'RightLeg', True),
    ('RightToeBase', 'RightToeBase', 'RightToeTip', 'RightFoot', True),
    ('Spine02', 'Spine02', 'Spine01', 'Hips', True),
    ('Spine01', 'Spine01', 'Spine', 'Spine02', True),
    ('Spine', 'Spine', 'neck', 'Spine01', True),
    ('LeftShoulder', 'LeftShoulder', 'LeftArm', 'Spine', True),
    ('LeftArm', 'LeftArm', 'LeftForeArm', 'LeftShoulder', True),
    ('LeftForeArm', 'LeftForeArm', 'LeftHand', 'LeftArm', True),
    ('LeftHand', 'LeftHand', 'LeftHandTip', 'LeftForeArm', True),
    ('RightShoulder', 'RightShoulder', 'RightArm', 'Spine', True),
    ('RightArm', 'RightArm', 'RightForeArm', 'RightShoulder', True),
    ('RightForeArm', 'RightForeArm', 'RightHand', 'RightArm', True),
    ('RightHand', 'RightHand', 'RightHandTip', 'RightForeArm', True),
    ('neck', 'neck', 'Head', 'Spine', True),
    ('Head', 'Head', 'head_end', 'neck', True),
    ('head_end', 'head_end', None, 'Head', False),
    ('headfront', 'headfront', None, 'Head', False),
]
X, Y = Vector((1, 0, 0)), Vector((0, 1, 0))


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args(sys.argv[sys.argv.index('--') + 1:])


def load_mesh(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(Path(path).resolve()))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if len(meshes) != 1:
        raise RuntimeError(f'{path}: expected one mesh, found {len(meshes)}')
    mesh = meshes[0]
    for o in bpy.context.scene.objects:
        o.select_set(o == mesh)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.context.scene.objects):
        if o != mesh:
            bpy.data.objects.remove(o, do_unlink=True)
    me = mesh.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    mn, mx = co.min(0), co.max(0)
    offset = np.array([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]])
    me.vertices.foreach_set('co', (co + offset).ravel())
    me.update()
    return mesh, offset


def build_armature(joints):
    J = {k: Vector(v) for k, v in joints.items()}
    missing = sorted({j for _, h, t, _, _ in BONES for j in (h, t) if j and j not in J})
    if missing:
        raise RuntimeError(f'spec joints missing: {missing}')
    arm = bpy.data.armatures.new('Armature')
    rig = bpy.data.objects.new('Armature', arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name, h, t, parent, deform in BONES:
        eb = arm.edit_bones.new(name)
        eb.head = J[h]
        if t:
            eb.tail = J[t]
        elif name == 'headfront':
            eb.tail = J[h] + (J[h] - J['Head']).normalized() * 0.05
        else:
            eb.tail = J[h] + (J[h] - J['Head']).normalized() * 0.05
        if (eb.tail - eb.head).length < 1e-4:
            raise RuntimeError(f'bone {name} has zero length')
        yax = (eb.tail - eb.head).normalized()
        ref = X if abs(yax.x) < 0.8 else Y
        xp = (ref - ref.dot(yax) * yax).normalized()
        eb.align_roll(xp.cross(yax))
        eb.use_deform = deform
    for name, _, _, parent, _ in BONES:
        if parent:
            arm.edit_bones[name].parent = arm.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig, J


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (a + ab * t - p).length


def heat_weights(mesh, rig):
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
    welded = len(dup.data.vertices)
    bpy.data.objects.remove(dup, do_unlink=True)
    return kd, dw, welded


def in_region(p, r, normal=None):
    if 'box' in r:
        (x0, y0, z0), (x1, y1, z1) = r['box']
        inside = x0 <= p.x <= x1 and y0 <= p.y <= y1 and z0 <= p.z <= z1
    elif 'capsule' in r:
        a, b, rad = r['capsule']
        a, b = Vector(a), Vector(b)
        inside = seg_dist(p, a, b) <= rad
        if inside and normal is not None and 'radialNormal' in r:
            ab = b - a
            t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
            radial = p - (a + ab * t)
            inside = radial.length > 1e-6 and normal.dot(radial.normalized()) >= r['radialNormal']
    else:
        raise RuntimeError(f'region needs box or capsule: {r}')
    w = r.get('where', {})
    for key, axis in (('x', 0), ('y', 1), ('z', 2)):
        if f'{key}Min' in w and p[axis] < w[f'{key}Min']:
            return False
        if f'{key}Max' in w and p[axis] > w[f'{key}Max']:
            return False
    return inside


def welded(me):
    """Welded vertex ids (the glTF import splits vertices along UV seams) and welded triangles."""
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    q = np.round(co / 1e-5).astype(np.int64)
    _, wid = np.unique(q, axis=0, return_inverse=True)
    wid = wid.ravel()
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get('vertices', tri)
    tri = wid[tri.reshape(-1, 3)]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    return co, wid, tri


def boundary_edges(me):
    """Welded edges used by exactly one triangle: open boundary (a hole, or a sheet's rim)."""
    _, _, tri = welded(me)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum())


def weld_graph(me):
    co, wid, tri = welded(me)
    nw = wid.max() + 1
    wco = np.zeros((nw, 3))
    wco[wid] = co
    fn = np.cross(wco[tri[:, 1]] - wco[tri[:, 0]], wco[tri[:, 2]] - wco[tri[:, 0]])
    vn = np.zeros((nw, 3))
    for k in range(3):
        np.add.at(vn, tri[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    e = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    adj = [[] for _ in range(nw)]
    for a, b in e:
        d = float(np.linalg.norm(wco[a] - wco[b]))
        adj[a].append((b, d))
        adj[b].append((a, d))
    return wid, wco, vn, e, adj


def grow_region(g, inside_w, dominant_w, spec):
    """Welded ids added to a rigid region: fold rings, then the down-facing underside."""
    from collections import deque
    wid, wco, vn, _, adj = g
    zmin = spec.get('zMin', -1e9)
    excl = set(spec.get('excludeDominant', []))
    ok = lambda w: wco[w][2] >= zmin and dominant_w[w] not in excl
    S = set(inside_w)
    front = set(inside_w)
    for _ in range(spec.get('foldRings', 0)):
        nxt = {w for v in front for w, _ in adj[v] if w not in S and ok(w)}
        S |= nxt
        front = nxt
    nz = spec.get('normalZMax')
    if nz is not None:
        dq = deque(S)
        while dq:
            v = dq.popleft()
            for w, _ in adj[v]:
                if w not in S and ok(w) and vn[w][2] < nz:
                    S.add(w)
                    dq.append(w)
    return S - set(inside_w)


def web_blend(g, Ww, fixed, sources, spec):
    """Harmonic blend of the welded weights Ww over the vertices within spec distance of sources."""
    import heapq
    wid, wco, vn, e, adj = g
    D = spec.get('distance', 0.1)
    dist = {v: 0.0 for v in sources}
    hq = [(0.0, v) for v in sources]
    while hq:
        d, v = heapq.heappop(hq)
        if d > dist.get(v, 1e9):
            continue
        for w, l in adj[v]:
            nd = d + l
            if nd <= D and nd < dist.get(w, 1e9) and not fixed[w]:
                dist[w] = nd
                heapq.heappush(hq, (nd, w))
    zone = np.array(sorted(v for v in dist if not fixed[v]), dtype=np.int64)
    if len(zone) == 0:
        return 0
    deg = np.zeros(len(Ww))
    np.add.at(deg, e[:, 0], 1)
    np.add.at(deg, e[:, 1], 1)
    for _ in range(spec.get('iterations', 400)):
        acc = np.zeros_like(Ww)
        np.add.at(acc, e[:, 0], Ww[e[:, 1]])
        np.add.at(acc, e[:, 1], Ww[e[:, 0]])
        Ww[zone] = acc[zone] / deg[zone, None]
    return len(zone)


def main():
    args = parse()
    spec = json.loads(Path(args.spec).read_text())
    rb = spec['rigBuild']
    mesh, offset = load_mesh(rb['mesh'])
    me = mesh.data
    for r in rb.get('regions', []):
        if 'detachFrom' in r:
            raise RuntimeError(f"region {r.get('note', '')!r}: detachFrom was removed (it deleted faces and opened "
                               'holes); use "web" to blend the fused faces instead')
    if 'detachPairs' in rb:
        raise RuntimeError('detachPairs was removed (it deleted faces and opened holes); use a region "web"')
    boundary_src = boundary_edges(me)
    rig, J = build_armature(rb['joints'])
    deform = [n for n, _, _, _, d in BONES if d]
    segs = {n: (J[h], J[t]) for n, h, t, _, d in BONES if d}
    kd, dw, welded = heat_weights(mesh, rig)

    per_vertex = []
    fallback = 0
    for v in me.vertices:
        _, idx, _ = kd.find(v.co)
        w = {k: x for k, x in dw[idx].items() if k in deform}
        if sum(w.values()) < 1e-3:
            w = {min(segs, key=lambda n: seg_dist(v.co, *segs[n])): 1.0}
            fallback += 1
        per_vertex.append(w)
    limited = {}
    for bone, lim in rb.get('boneLimits', {}).items():
        n = 0
        for v, w in zip(me.vertices, per_vertex):
            if bone in w and not in_region(v.co, {'box': [[-1e9] * 3, [1e9] * 3], 'where': lim}):
                del w[bone]
                n += 1
                if not w:
                    w[min((k for k in segs if k != bone), key=lambda k: seg_dist(v.co, *segs[k]))] = 1.0
        limited[bone] = n
    region_counts = []
    for r in rb.get('regions', []):
        n = 0
        for v in me.vertices:
            if in_region(v.co, r, v.normal):
                per_vertex[v.index] = dict(r['weights'])
                n += 1
        region_counts.append({'note': r.get('note', ''), 'verts': n, 'inside': {v.index for v in me.vertices if in_region(v.co, r, v.normal)}})
        if n == 0:
            raise RuntimeError(f'region matched no vertices (wrong coordinates?): {r}')

    wid, wco, vn, e, adj = g = weld_graph(me)
    nw = wid.max() + 1
    bone_ix = {n: i for i, n in enumerate(deform)}
    Ww = np.zeros((nw, len(deform)))
    for v, w in zip(me.vertices, per_vertex):
        row = np.zeros(len(deform))
        for k, x in w.items():
            row[bone_ix[k]] = x
        Ww[wid[v.index]] = row / max(row.sum(), 1e-9)
    heat_dom = [deform[i] for i in Ww.argmax(1)]
    fixed = np.zeros(nw, bool)
    region_w = []
    grown, webbed = [], []
    for r, rc in zip(rb.get('regions', []), region_counts):
        inside_w = {wid[i] for i in rc.pop('inside')}
        row = np.zeros(len(deform))
        for k, x in r['weights'].items():
            row[bone_ix[k]] = x
        if 'grow' in r:
            add = grow_region(g, inside_w, heat_dom, r['grow'])
            grown.append({'note': r.get('note', ''), 'addedVerts': len(add)})
            inside_w |= add
        for w in inside_w:
            Ww[w] = row
        fixed[list(inside_w)] = True
        region_w.append(inside_w)
    for r, inside_w in zip(rb.get('regions', []), region_w):
        if 'web' in r:
            n = web_blend(g, Ww, fixed, inside_w, r['web'])
            webbed.append({'note': r.get('note', ''), 'blendedVerts': n})
    per_vertex = [{deform[k]: float(x) for k, x in enumerate(Ww[wid[v.index]]) if x > 1e-4} for v in me.vertices]
    for gr in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(gr)
    groups = {n: mesh.vertex_groups.new(name=n) for n in deform}
    counts = {n: 0 for n in deform}
    for v, w in zip(me.vertices, per_vertex):
        top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
        s_ = sum(x for _, x in top)
        for n, x in top:
            if x / s_ > 1e-3:
                groups[n].add([v.index], x / s_, 'REPLACE')
                counts[n] += 1
    boundary_out = boundary_edges(me)
    if boundary_out > boundary_src:
        raise RuntimeError(f'open boundary grew: {boundary_src} welded boundary edges in the source mesh, '
                           f'{boundary_out} after rigging. Faces must never be deleted (see the docstring).')
    mesh.parent = rig
    for m in list(mesh.modifiers):
        mesh.modifiers.remove(m)
    mod = mesh.modifiers.new('Armature', 'ARMATURE')
    mod.object = rig
    mesh.name = spec.get('unit', 'character')

    sums = [sum(g.weight for g in v.groups) for v in me.vertices]
    infl = [len(v.groups) for v in me.vertices]
    report = {
        'spec': args.spec, 'mesh': rb['mesh'], 'normaliseOffset': [round(float(x), 5) for x in offset],
        'height': round(max(v.co.z for v in me.vertices), 4), 'verts': len(me.vertices), 'weldedVerts': welded,
        'heatFallbackVerts': fallback, 'boneLimitedVerts': limited, 'regions': region_counts, 'grown': grown, 'webs': webbed,
        'boundaryEdges': {'source': boundary_src, 'rigged': boundary_out}, 'boneVertexCounts': counts,
        'emptyBones': [n for n, c in counts.items() if c == 0],
        'skin': {'weightSumMin': round(min(sums), 5), 'weightSumMax': round(max(sums), 5),
                 'unweighted': sum(1 for s in sums if s < 1e-3), 'maxInfluences': max(infl)},
        'joints': {k: [round(c, 4) for c in v] for k, v in rb['joints'].items()},
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    for o in bpy.context.scene.objects:
        o.select_set(o in (mesh, rig))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True,
                              export_animations=False, export_skins=True, export_all_influences=False,
                              export_yup=True, export_apply=False)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RIGGED', json.dumps({k: report[k] for k in ('height', 'verts', 'heatFallbackVerts', 'regions', 'grown', 'webs',
                                                        'boundaryEdges', 'emptyBones', 'skin')}))


if __name__ == '__main__':
    main()
