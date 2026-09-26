# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Inspect a static Meshy critter mesh before rigging (workflow step 1).

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/prep_critter.py -- \
  --input input/cow.glb --out-dir output/critter-authoring/cow/prep

Writes, in rig space (the mesh translated so its lowest point is z = 0 and its
bounding box is centred in x and y; scale and orientation untouched, so the
head faces -Y, which is glTF +Z, the game's "front"):
  verts.npy          float32 (N, 3) vertex positions
  sdf.npy            float32 (N,) shape diameter (local thickness) per vertex
  faces.npy          int32   (M, 3) triangle vertex indices
  views.json         orthographic camera params for side/front/top + the offset
  side.png front.png top.png   textured Workbench renders from those cameras
Side looks from +X (the animal's LEFT; head on the image left), front looks
from -Y, top looks down with the head at the image top. project() in
fit_landmarks.py maps rig-space points to these images.
"""
import bpy, bmesh, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
inp, out = opt['input'], opt['out-dir']
size = int(opt.get('size', 768))
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=inp)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
assert len(meshes) == 1, f'expected one mesh, got {len(meshes)}'
ob = meshes[0]
# Bake the import transform (Y-up -> Z-up parent rotation) into the vertices.
for o in bpy.context.scene.objects:
    o.select_set(o == ob)
bpy.context.view_layer.objects.active = ob
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

me = ob.data
co = np.empty(len(me.vertices) * 3, dtype=np.float32)
me.vertices.foreach_get('co', co)
co = co.reshape(-1, 3)
# Optional yaw correction (degrees about +Z) for a Meshy mesh whose body axis
# is not aligned with Y: a critter whose body is yawed would crab sideways
# when the game moves it along its heading. Applied before centring.
yaw = float(opt.get('yaw', 0.0))
pitch = float(opt.get('pitch', 0.0))
from rig_space import rig_rotation  # noqa: E402
Rm = rig_rotation(yaw, pitch)
co = co @ Rm
mn, mx = co.min(0), co.max(0)
offset = np.array([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]], dtype=np.float32)
cx = opt.get('center-x', 'bbox')
if cx not in ('bbox', ''):
    # centre x on the BODY, not the bounding box (a tail curled to one side
    # skews the box): 'core' = the median x of the thickest third of the
    # mesh by slab width, or a number
    if cx == 'core':
        zc = co[:, 2] > np.percentile(co[:, 2], 40)
        offset[0] = -float(np.median(co[zc][:, 0]))
    else:
        offset[0] = -float(cx)
co += offset
me.vertices.foreach_set('co', co.ravel())
me.update()

bm = bmesh.new(); bm.from_mesh(me)
bmesh.ops.triangulate(bm, faces=bm.faces[:])
faces = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int32)
non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
bm.free()
np.save(os.path.join(out, 'verts.npy'), co)
np.save(os.path.join(out, 'faces.npy'), faces)

# Shape diameter per vertex (sdf.npy): rays cast INTO the mesh (against the
# normal, in a 25-degree cone) to the opposite wall, median hit distance. A
# leg or claw arm is thin, a carapace or abdomen thick, so fit_arthropod.py
# uses it to find where a limb meets the body. Computed on a welded copy:
# Meshy splits vertices on UV seams and split normals point along the seams.
from mathutils.bvhtree import BVHTree
from mathutils import kdtree
bmw = bmesh.new(); bmw.from_mesh(me)
bmesh.ops.remove_doubles(bmw, verts=bmw.verts, dist=1e-4)
bmw.normal_update()
bvh = BVHTree.FromBMesh(bmw)
wverts = [(v.co.copy(), v.normal.copy()) for v in bmw.verts]
bmw.free()
span_ = float(np.ptp(co, axis=0).max())
cone = [Vector((0, 0, 1))] + [Vector((float(np.sin(0.44) * np.cos(a)), float(np.sin(0.44) * np.sin(a)), float(np.cos(0.44))))
                              for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
wsdf = np.full(len(wverts), span_, dtype=np.float32)
for i, (p, nrm) in enumerate(wverts):
    if nrm.length < 1e-6:
        continue
    n = -nrm
    rot = Vector((0, 0, 1)).rotation_difference(n)
    o = p + n * 1e-4
    hits = [h[3] for h in (bvh.ray_cast(o, rot @ c, span_) for c in cone) if h[0] is not None]
    if hits:
        wsdf[i] = float(np.median(hits))
kd = kdtree.KDTree(len(wverts))
for i, (p, _) in enumerate(wverts):
    kd.insert(p, i)
kd.balance()
sdf = np.array([wsdf[kd.find(Vector(c.tolist()))[1]] for c in co], dtype=np.float32)
np.save(os.path.join(out, 'sdf.npy'), sdf)

mn, mx = co.min(0), co.max(0)
center = (mn + mx) / 2
span = float((mx - mn).max()) * 1.12

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
sh = scene.display.shading
sh.light = 'STUDIO'; sh.color_type = 'TEXTURE'
sh.show_cavity = False; sh.show_shadows = False
scene.display.render_aa = '8'
scene.render.resolution_x = scene.render.resolution_y = size
w = bpy.data.worlds.new('W'); scene.world = w; w.color = (0.62, 0.64, 0.6)
scene.view_settings.view_transform = 'Standard'
cd = bpy.data.cameras.new('C'); cd.type = 'ORTHO'; cd.ortho_scale = span
cam = bpy.data.objects.new('C', cd); scene.collection.objects.link(cam); scene.camera = cam
cd.clip_start = 0.001; cd.clip_end = 100

views = {}
c = Vector(center.tolist())
for name, d, up in (
    ('side', Vector((1, 0, 0)), Vector((0, 0, 1))),
    ('front', Vector((0, -1, 0)), Vector((0, 0, 1))),
    ('top', Vector((0, 0, 1)), Vector((0, -1, 0))),
):
    cam.location = c + d * span * 3
    look = -d
    right = look.cross(up).normalized()
    true_up = right.cross(look).normalized()
    # Build the camera rotation from (right, up, -look) columns.
    from mathutils import Matrix
    m = Matrix((right, true_up, -look)).transposed()
    cam.rotation_euler = m.to_euler()
    scene.render.filepath = os.path.join(out, f'{name}.png')
    bpy.ops.render.render(write_still=True)
    views[name] = {'center': list(c), 'right': list(right), 'up': list(true_up),
                   'ortho_scale': span, 'res': size}

info = {
    'input': inp, 'yaw_deg': yaw, 'pitch_deg': pitch, 'offset': offset.tolist(), 'min': mn.tolist(), 'max': mx.tolist(),
    'verts': int(len(co)), 'tris': int(len(faces)), 'non_manifold_edges': int(non_manifold),
    'views': views,
}
json.dump(info, open(os.path.join(out, 'views.json'), 'w'), indent=1)
print('PREP ' + json.dumps({k: info[k] for k in ('verts', 'tris', 'non_manifold_edges', 'min', 'max')}))
