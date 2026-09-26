# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Which bone pushes a clip through the ground? (diagnostic)

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/diag_ground.py -- --input <character.glb> --clip walk

For every frame of the clip, finds the lowest deformed vertex and reports the
worst frames with that vertex's dominant bone and its rest position.
"""
import bpy, sys
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = 30
bpy.ops.import_scene.gltf(filepath=opt['input'])
rig = [o for o in sc.objects if o.type == 'ARMATURE'][0]
mesh = [o for o in sc.objects if o.type == 'MESH'][0]
act = bpy.data.actions[opt.get('clip', 'walk')]
rig.animation_data.action = act
rig.animation_data.action_slot = act.slots[0]
gn = {g.index: g.name for g in mesh.vertex_groups}
M = np.array(mesh.matrix_world)
rest = np.array([M[:3, :3] @ np.array(v.co) + M[:3, 3] for v in mesh.data.vertices])
dg = bpy.context.evaluated_depsgraph_get()
rows = []
a, b = [int(round(x)) for x in act.frame_range]
for f in range(a, b + 1):
    sc.frame_set(f)
    ev = mesh.evaluated_get(dg); m = ev.to_mesh()
    arr = np.empty(len(m.vertices) * 3); m.vertices.foreach_get('co', arr); ev.to_mesh_clear()
    arr = arr.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    i = int(arr[:, 2].argmin())
    v = mesh.data.vertices[i]
    dom = max(v.groups, key=lambda g: g.weight)
    rows.append((float(arr[i, 2]), f, gn[dom.group], round(dom.weight, 2), rest[i].round(3).tolist()))
rows.sort()
for r in rows[:6]:
    print('LOW z=%.4f frame=%d bone=%s w=%.2f rest=%s' % r)
