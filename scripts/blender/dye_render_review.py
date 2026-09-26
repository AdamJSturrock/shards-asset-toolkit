# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Blender: render a dye-masked unit GLB with its colour-coded mask.

    blender -b --factory-startup -P scripts/blender/dye_render_review.py -- \
        <masked.glb> <overlay.png> <out.png>

<overlay.png> is the review overlay shards-asset dye-mask writes (base dimmed, accent
magenta, extended cyan, in the GLB's UV space). Every material's colour is
swapped for it and the model is rendered from four sides (front, right, back,
left) with the Workbench engine, flat texture colour, into
<out>_v0.png … <out>_v3.png, which shards-asset dye-mask stitches into <out.png>.

Bind pose (no clip is applied). Pixel review only: the GLB is never written.
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb_path, overlay_path, out_path = argv[0], argv[1], argv[2]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

overlay = bpy.data.images.load(overlay_path)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in " + glb_path)

for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = overlay
    tex.interpolation = "Closest"
    nodes.active = tex

# Frame the bind pose.
lo = Vector((1e9, 1e9, 1e9))
hi = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
        hi = Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
centre = (lo + hi) / 2
size = max((hi - lo).x, (hi - lo).y, (hi - lo).z)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "TEXTURE"
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("review") if scene.world is None else scene.world

cam_data = bpy.data.cameras.new("review")
cam_data.type = "ORTHO"
cam_data.ortho_scale = size * 1.15
cam = bpy.data.objects.new("review", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

base = out_path[:-4] if out_path.lower().endswith(".png") else out_path
for i in range(4):
    yaw = i * math.pi / 2
    d = size * 2.5
    # glTF +Z (front) imports as Blender -Y.
    cam.location = centre + Vector((math.sin(yaw) * d, -math.cos(yaw) * d, size * 0.15))
    direction = centre - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{base}_v{i}.png"
    bpy.ops.render.render(write_still=True)
print("review views written:", base + "_v[0-3].png")
