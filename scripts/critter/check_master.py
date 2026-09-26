# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Reopen a saved critter master and check ordinary playback (workflow
acceptance item 3: render scripts do not prove the saved file plays).

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/check_master.py -- --blend <out>/<critter>.blend

For every action: assign it to the rig in the reopened file, step through its
frames with the scene's own playback settings, and confirm the skinned mesh
actually moves (a positive control: an action whose mesh never leaves the
rest pose fails). Also lists the pose markers (event names and frames).
Exits non-zero on any failure.
"""
import bpy, sys, json
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
bpy.ops.wm.open_mainfile(filepath=opt['blend'])
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == 'ARMATURE')
mesh = next(o for o in sc.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers))
dg = bpy.context.evaluated_depsgraph_get()


def verts():
    dg.update()
    ev = mesh.evaluated_get(dg)
    m = ev.to_mesh()
    a = np.empty(len(m.vertices) * 3)
    m.vertices.foreach_get('co', a)
    ev.to_mesh_clear()
    return a.reshape(-1, 3)


out, bad = {}, []
for act in bpy.data.actions:
    rig.animation_data.action = act
    if act.slots:
        rig.animation_data.action_slot = act.slots[0]
    a, b = (int(round(x)) for x in act.frame_range)
    sc.frame_set(a)
    v0 = verts()
    moved = 0.0
    for f in range(a, b + 1):
        sc.frame_set(f)
        moved = max(moved, float(np.abs(verts() - v0).max()))
    span = float(np.ptp(v0, axis=0).max())
    ok = moved > 1e-3 * span
    out[act.name] = {'frames': [a, b], 'max_vertex_motion_rel': round(moved / span, 4),
                     'markers': {m.name: m.frame for m in act.pose_markers}, 'ok': ok}
    if not ok:
        bad.append(act.name)
print('MASTER ' + json.dumps({'blend': opt['blend'], 'fps': sc.render.fps, 'actions': out}))
if bad:
    raise SystemExit(f'actions that do not move the mesh: {bad}')
