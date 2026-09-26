# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Copy a labelled limb to make a missing one (Meshy dropped a leg).

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/add_limb.py -- --critter <profile> \
  --copy leg2.L,leg2.R [--mirror 1] [--yaw 25] [--move dx,dy,dz] \
  [--grow 3] [--sink-box x0,x1,y0,y1,z0,z1 --abdomen-y Y] --out <new.glb>

Reads the profile's source GLB and the fit's per-vertex limb labels
(<work>/<critter>/prep/labels.npy, same vertex order as the import), takes
every face of the `--copy` limb plus `--grow` rings of the body around its
root (so the copy's base sinks into the body and the join stays hidden),
and duplicates them with their UVs and material:
  --copy       one or more labels; each is copied on its own; a .R copy
               mirrors the placement (yaw and the x of --move flip sign)
  --mirror 1   reflect across x = 0 (a left leg becomes a right one; the
               face winding is flipped so normals stay outward)
  --yaw deg    turn the copy about the vertical through the limb's root
               (the root = centroid of the copied body rings)
  --move       then translate it (rig space: head -Y, left +X, up +Z)
The result is written as a new GLB (rig space, texture kept) for prep ->
fit -> rig; the source file is never touched. Prints the copied vertex
count and the root. LOOK at the before/after render (make it with
render_labels.py or prep's views) before rigging.
"""
import bpy, bmesh, sys, os, json, math
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import work_dir, source_glb  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
NAME = opt['critter']
from paths import load_profiles  # noqa: E402
PROF = load_profiles()[NAME]
prep = work_dir(NAME, 'prep')
views = json.load(open(os.path.join(prep, 'views.json')))
labels = np.load(os.path.join(prep, 'labels.npy'))
names = json.load(open(os.path.join(prep, 'labels.json')))['names']

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source_glb(NAME, PROF))
mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
for o in bpy.context.scene.objects:
    o.select_set(o == mesh)
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(bpy.context.scene.objects):
    if o != mesh:
        bpy.data.objects.remove(o, do_unlink=True)
me = mesh.data
assert len(me.vertices) == len(labels), f'labels {len(labels)} != verts {len(me.vertices)}: re-run prep + fit'
# rig space exactly as prep: optional yaw, then prep's centring offset
co = np.empty(len(me.vertices) * 3, dtype=np.float32)
me.vertices.foreach_get('co', co)
co = co.reshape(-1, 3)
from rig_space import rig_rotation  # noqa: E402
co = co @ rig_rotation(float(views.get('yaw_deg', 0.0)), float(views.get('pitch_deg', 0.0)))
co += np.array(views['offset'], dtype=np.float32)
me.vertices.foreach_set('co', co.ravel())
me.update()

bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
pos_of = {}
for v in bm.verts:
    pos_of.setdefault(tuple(np.round(np.array(v.co), 5)), []).append(v.index)
done = []
for copy in [c for c in opt.get('copy', '').split(',') if c]:
    # a .R copy of a .L instruction mirrors the placement: yaw and x move flip
    right = copy.endswith('.R')
    src_label = names.index(copy)
    bm.verts.ensure_lookup_table()
    sel = {v.index for v in bm.verts if v.index < len(labels) and labels[v.index] == src_label}
    limb_only = set(sel)
    # weld-aware ring growth: glTF splits vertices on UV seams, so grow by
    # position as well as by edges
    for _ in range(int(opt.get('grow', 3))):
        add = set()
        for i in sel:
            v = bm.verts[i]
            for e in v.link_edges:
                add.add(e.other_vert(v).index)
        for i in list(add):
            add.update(pos_of.get(tuple(np.round(np.array(bm.verts[i].co), 5)), []))
        sel |= add
    root_ring = sel - limb_only
    root = Vector(np.mean([np.array(bm.verts[i].co) for i in (root_ring or sel)], axis=0).tolist())
    faces = [f for f in bm.faces if all(v.index in sel for v in f.verts)]
    dup = bmesh.ops.duplicate(bm, geom=faces)
    new_verts = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMVert)]
    new_faces = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMFace)]
    M = Matrix.Identity(4)
    root_m = root.copy()
    if opt.get('mirror') == '1':
        M = Matrix.Scale(-1, 4, Vector((1, 0, 0))) @ M
        root_m = Vector((-root.x, root.y, root.z))
    yaw_ = float(opt.get('yaw', 0)) * (-1 if right else 1)
    if yaw_:
        R = Matrix.Rotation(math.radians(yaw_), 4, 'Z')
        M = Matrix.Translation(root_m) @ R @ Matrix.Translation(-root_m) @ M
    mv = Vector([float(x) for x in opt.get('move', '0,0,0').split(',')])
    if right:
        mv.x = -mv.x
    M = Matrix.Translation(mv) @ M
    bmesh.ops.transform(bm, matrix=M, verts=new_verts)
    if opt.get('mirror') == '1':
        bmesh.ops.reverse_faces(bm, faces=new_faces, flip_multires=False)
    done.append({'copied': copy, 'limb_verts': len(limb_only), 'with_root_rings': len(sel),
                 'new_faces': len(new_faces), 'root': [round(x, 3) for x in root], 'yaw': yaw_,
                 'move': [round(x, 3) for x in mv]})
if 'sink-box' in opt:
    # SINK a fused, unwanted limb into the abdomen: every vertex inside the
    # box that lies outside the abdomen's ellipsoid is pulled radially to
    # just inside it (so nothing of the old leg is left showing). The
    # ellipsoid is the bounding ellipsoid (1st..99th percentile) of the
    # body vertices behind `abdomen-y` that are not in the box.
    x0, x1, y0, y1, z0, z1 = (float(t) for t in opt['sink-box'].split(','))
    bm.verts.ensure_lookup_table()
    P = np.array([v.co[:] for v in bm.verts[:len(labels)]])   # the ORIGINAL vertices only
    inbox = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > z0) & (P[:, 2] < z1)
    sdf = np.load(os.path.join(prep, 'sdf.npy'))
    body = (labels[:len(P)] == -1) & (P[:, 1] > float(opt.get('abdomen-y', 0.0))) & ~inbox & (sdf > 0.3 * sdf.max())
    lo, hi = np.percentile(P[body], 1, axis=0), np.percentile(P[body], 99, axis=0)
    cen, rad = (lo + hi) / 2, (hi - lo) / 2
    cen[0] = 0.0                                   # the abdomen is symmetric
    rad[0] = float(np.percentile(np.abs(P[body][:, 0]), 99))
    q = (P - cen) / rad
    rr = np.linalg.norm(q, axis=1)
    # only THIN geometry moves (the leg; shape diameter under `sink-sdf` of
    # the thickest), never the abdomen's own surface under it
    thin = sdf < float(opt.get('sink-sdf', 0.3)) * sdf.max()
    # never a vertex some limb label owns: only the body-labelled fused leg
    move = inbox & thin & (rr > 0.5) & (labels[:len(P)] == -1)
    # push each thin vertex toward the abdomen's centre until it is just
    # behind the abdomen's REAL surface (a ray against the untouched faces),
    # not onto the ellipsoid: the abdomen is not an ellipsoid near its waist
    from mathutils.bvhtree import BVHTree
    keep_faces = [f for f in bm.faces if not any(v.index < len(move) and move[v.index] for v in f.verts)]
    kb = bmesh.new()
    vmap = {}
    for f in keep_faces:
        vs = []
        for v in f.verts:
            if v.index not in vmap:
                vmap[v.index] = kb.verts.new(v.co)
            vs.append(vmap[v.index])
        try:
            kb.faces.new(vs)
        except ValueError:
            pass
    kb.verts.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(kb)
    depth = float(opt.get('sink-depth', 0.03))
    missed = 0
    def hits(p, d):
        n, o = 0, p.copy()
        for _ in range(64):
            loc, nrm, idx, dd = bvh.ray_cast(o, d)
            if loc is None:
                break
            n += 1
            o = loc + d * 1e-4
        return n

    def inside(p):
        # ray parity, majority of three directions (robust to seams), and
        # `depth` clear of the surface
        loc, nrm, idx, dd = bvh.find_nearest(p)
        if loc is None or dd < depth:
            return False
        votes = sum(hits(p, Vector(d)) % 2 for d in ((0, 0, 1), (0.6, 0.3, 0.74), (-0.5, 0.7, 0.5)))
        return votes >= 2

    for i in np.flatnonzero(move):
        p0 = Vector(P[i].tolist())
        d = Vector(cen.tolist()) - p0
        dist = d.length
        d.normalize()
        # march toward the abdomen centre until the point is inside the
        # untouched surface by `depth` (nearest-face normal test)
        t, placed = 0.0, False
        while t < dist:
            p = p0 + d * t
            if inside(p):
                bm.verts[int(i)].co = p
                placed = True
                break
            t += 0.01
        if not placed:
            bm.verts[int(i)].co = Vector(cen.tolist())
            missed += 1
    # The leg's shape diameter is not everywhere thin (a ray from its inner
    # face can cross into the abdomen): any still-outside box vertex most of
    # whose neighbours were sunk belonged to the leg too; sink it and repeat.
    moved = set(np.flatnonzero(move).tolist())
    for _ in range(int(opt.get('sink-spread', 4))):
        more = []
        for i in np.flatnonzero(inbox & (labels[:len(P)] == -1)):
            if i in moved:
                continue
            v = bm.verts[int(i)]
            nb = [e.other_vert(v).index for e in v.link_edges]
            if nb and sum(n in moved for n in nb) / len(nb) >= 0.5 and not inside(v.co):
                more.append(int(i))
        for i in more:
            p0 = bm.verts[i].co.copy()
            d = (Vector(cen.tolist()) - p0)
            dist = d.length
            d.normalize()
            t = 0.0
            while t < dist and not inside(p0 + d * t):
                t += 0.01
            bm.verts[i].co = p0 + d * min(t, dist)
            moved.add(i)
        if not more:
            break
    kb.free()
    done.append({'sunk_vertices': len(moved), 'sink_rays_missed': missed, 'abdomen_centre': cen.round(3).tolist(),
                 'abdomen_radii': rad.round(3).tolist()})
bm.to_mesh(me)
me.update()
# keep the imported custom normals out of the new faces' way: recompute
if hasattr(me, 'free_normals_split'):
    me.free_normals_split()
bm.free()
out = os.path.abspath(os.path.expanduser(opt['out']))
os.makedirs(os.path.dirname(out), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', use_selection=True, export_yup=True,
                          export_image_format='AUTO')
print('ADD_LIMB ' + json.dumps({'copies': done, 'mirror': opt.get('mirror') == '1', 'out': out}))
