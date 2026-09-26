# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Validate and render an exported critter GLB (workflow steps 4-6).

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/review_rig.py -- \
  --input output/critter-authoring/cow/v1/character.glb --out-dir output/critter-authoring/cow/v1/review

Works on the EXPORTED file (clean re-import), not the .blend. Writes:
  validation.json  clips + durations, skin normalisation, influence counts,
                   loop-seam error, ground contact per clip, finiteness
  weights.png      actual vertex weights: each vertex tinted by its bones'
                   colours (weight-blended), side / front / top / 3-4 views
  clip_<name>.png  sampled frames of each clip, textured, side + 3/4 view
  preview.mp4      with --video 1: EVERY frame of every clip from a game-like
                   3/4 camera (idle, walk looped twice, variants, death),
                   because a sampled strip cannot show a seam or a pop
Numbers are evidence, not approval: look at the renders.
"""
import bpy, sys, os, json, math, colorsys
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
OUT = os.path.abspath(opt['out-dir'])
os.makedirs(OUT, exist_ok=True)
SIZE = int(opt.get('size', 240))
NFR = int(opt.get('frames', 8))

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
# BEFORE importing: the glTF importer converts clip seconds to frames at the
# scene rate, and the factory default is 24 fps (the trap the workflow
# records: a 24-fps read of a 30-fps export silently shortens every clip).
sc.render.fps = 30
bpy.ops.import_scene.gltf(filepath=opt['input'])
rig = [o for o in sc.objects if o.type == 'ARMATURE'][0]
mesh = [o for o in sc.objects if o.type == 'MESH'][0]
actions = sorted(bpy.data.actions, key=lambda a: a.name)
dg = bpy.context.evaluated_depsgraph_get()


def set_action(act):
    rig.animation_data.action = act
    if act.slots:
        rig.animation_data.action_slot = act.slots[0]


def frame_range(act):
    a, b = act.frame_range
    return int(round(a)), int(round(b))


def eval_verts():
    dg.update()
    ev = mesh.evaluated_get(dg)
    m = ev.to_mesh()
    arr = np.empty(len(m.vertices) * 3, dtype=np.float64)
    m.vertices.foreach_get('co', arr)
    ev.to_mesh_clear()
    arr = arr.reshape(-1, 3)
    M = np.array(mesh.matrix_world)
    return arr @ M[:3, :3].T + M[:3, 3]


# ── validation ─────────────────────────────────────────────────────────────
rep = {'input': opt['input'], 'clips': {}, 'bones': len(rig.data.bones)}
gnames = {g.index: g.name for g in mesh.vertex_groups}
sums, infl, bad = [], [], 0
for v in mesh.data.vertices:
    ws = [g.weight for g in v.groups if g.weight > 1e-5]
    sums.append(sum(ws)); infl.append(len(ws))
sums = np.array(sums); infl = np.array(infl)
rep['skin'] = {'verts': int(len(sums)), 'weight_sum_min': float(sums.min()), 'weight_sum_max': float(sums.max()),
               'unweighted': int((sums < 1e-3).sum()), 'max_influences': int(infl.max())}
rest = None
for act in actions:
    set_action(act)
    a, b = frame_range(act)
    mins, fin = [], True
    first = last = None
    for f in range(a, b + 1):
        sc.frame_set(f)
        vv = eval_verts()
        fin &= bool(np.isfinite(vv).all())
        mins.append(float(vv[:, 2].min()))
        if f == a:
            first = vv
        last = vv
    span = float(np.ptp(first, axis=0).max())
    seam = float(np.abs(last - first).max()) / span
    rep['clips'][act.name] = {'frames': b - a + 1, 'seconds': round((b - a) / 30.0, 3),
                              'min_z_lowest': round(min(mins), 4), 'min_z_last': round(mins[-1], 4),
                              'loop_seam_rel': round(seam, 5), 'finite': fin}
# ── foot plant (walk clips) ────────────────────────────────────────────────
# Every foot on the ground must move with the ground: in an in-place walk the
# planted feet all slide back at the SAME velocity (the travel speed). A foot
# whose velocity differs from the others' while it is down is skating.
# Feet = the lowest vertex of each rest-pose ground contact cluster.
set_action(actions[0]); sc.frame_set(frame_range(actions[0])[0])
for act in actions:
    if act.name != 'walk':
        continue
    set_action(act)
    a, b = frame_range(act)
    sc.frame_set(a)
    v0w = eval_verts()
    Hh = float(v0w[:, 2].max())
    low = np.flatnonzero(v0w[:, 2] < v0w[:, 2].min() + 0.05 * Hh)
    rad = 0.04 * float(np.ptp(v0w[:, :2], axis=0).max())
    feet, left = [], list(low)
    while left:
        i = left[0]
        grp = [j for j in left if np.linalg.norm(v0w[j, :2] - v0w[i, :2]) < rad]
        feet.append(min(grp, key=lambda j: v0w[j, 2]))
        left = [j for j in left if j not in grp]
    P = []
    for f in range(a, b + 1):
        sc.frame_set(f)
        P.append(eval_verts()[feet])
    P = np.array(P)                              # frames x feet x 3
    # down = within a tenth of the foot's own lift above its lowest point in
    # the clip (scale-free: a foot just lifting or landing moves with the
    # swing, it is not planted)
    base = P[:, :, 2].min(0)
    thr = np.maximum(0.1 * (P[:, :, 2].max(0) - base), 1e-4 * Hh)
    down = (P[:-1, :, 2] - base < thr) & (P[1:, :, 2] - base < thr)
    V = P[1:, :, :2] - P[:-1, :, :2]
    slides, speeds = [], []
    for k in range(len(V)):
        idx = np.flatnonzero(down[k])
        if len(idx) < 2:
            continue
        g = np.median(V[k, idx], axis=0)
        speeds.append(float(np.linalg.norm(g)))
        slides += [float(np.linalg.norm(V[k, j] - g)) for j in idx]
    sp = float(np.median(speeds)) if speeds else 0.0
    rep['clips']['walk']['plant'] = {
        'feet': len(feet), 'ground_speed_per_frame': round(sp, 5),
        'min_feet_down': int(down.sum(1).min()),
        'slide_max_per_frame': round(max(slides), 5) if slides else None,
        'slide_rel_to_speed': round(max(slides) / sp, 3) if slides and sp > 0 else None}
# The same test on the FOOT BONES (each leg's last bone's tail = the IK
# target) when the rig has them (the traced plans' 'legN_tarsus.*'): the
# mesh-contact version above also counts a belly, a claw or a tail lying on
# the ground as feet. Planted = within 2% of the foot's own lift of its
# lowest point; `min_feet_down` 0 means no frame has two feet exactly down
# (the body was lifted off its feet somewhere, e.g. by the ground solve).
fb = [b.name for b in rig.pose.bones if 'tarsus' in b.name]
for act in actions:
    if act.name != 'walk' or not fb:
        continue
    set_action(act)
    a, b = frame_range(act)
    P = []
    for f in range(a, b + 1):
        sc.frame_set(f)
        P.append([np.array(rig.matrix_world @ rig.pose.bones[n].tail) for n in fb])
    P = np.array(P)
    z = P[:, :, 2]; base = z.min(0); lift = np.maximum(z.max(0) - base, 1e-6)
    down = (z[:-1] - base < 0.02 * lift) & (z[1:] - base < 0.02 * lift)
    V = P[1:, :, :2] - P[:-1, :, :2]
    sl, sp = [], []
    for k in range(len(V)):
        idx = np.flatnonzero(down[k])
        if len(idx) < 2:
            continue
        g = np.median(V[k, idx], axis=0)
        sp.append(float(np.linalg.norm(g)))
        sl += [float(np.linalg.norm(V[k, j] - g)) for j in idx]
    s_ = float(np.median(sp)) if sp else 0.0
    rep['clips']['walk']['plant_bones'] = {
        'feet': len(fb), 'min_feet_down': int(down.sum(1).min()), 'ground_speed_per_frame': round(s_, 5),
        'slide_rel_to_speed': round(max(sl) / s_, 3) if sl and s_ > 0 else None,
        'foot_lift': round(float(np.median(lift)), 4)}
json.dump(rep, open(os.path.join(OUT, 'validation.json'), 'w'), indent=1)
print('VALIDATION ' + json.dumps(rep))

# ── render setup ───────────────────────────────────────────────────────────
sc.render.engine = 'BLENDER_WORKBENCH'
sh = sc.display.shading
sh.light = 'STUDIO'; sh.show_cavity = False; sh.show_shadows = False
sc.display.render_aa = '8'
sc.render.resolution_x = sc.render.resolution_y = SIZE
w = bpy.data.worlds.new('W'); sc.world = w; w.color = (0.62, 0.64, 0.6)
sc.view_settings.view_transform = 'Standard'
cd = bpy.data.cameras.new('C'); cd.type = 'ORTHO'
cam = bpy.data.objects.new('C', cd); sc.collection.objects.link(cam); sc.camera = cam
set_action(actions[0]); sc.frame_set(frame_range(actions[0])[0])
v0 = eval_verts()
C = Vector(((v0[:, 0].min() + v0[:, 0].max()) / 2, (v0[:, 1].min() + v0[:, 1].max()) / 2, (v0[:, 2].max()) / 2))
R = float(np.ptp(v0, axis=0).max())
VIEWS = {'side': (Vector((1, 0, 0.0001)), 1.35), 'front': (Vector((0, -1, 0.0001)), 1.35),
         'top': (Vector((0, 0, 1)), 1.35), 'q34': (Vector((1, -1, 0.9)), 1.45)}


def aim(view):
    d, scale = VIEWS[view]
    d = d.normalized()
    cam.location = C + d * R * 4
    look = -d
    up = Vector((0, 0, 1)) if abs(look.z) < 0.99 else Vector((0, -1, 0))
    right = look.cross(up).normalized(); true_up = right.cross(look)
    cam.rotation_euler = Matrix((right, true_up, -look)).transposed().to_euler()
    cd.ortho_scale = R * scale
    cd.clip_start, cd.clip_end = 0.001, R * 20


def render(path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def compose(tiles, cols, path):
    rows = (len(tiles) + cols - 1) // cols
    W, H = SIZE * cols, SIZE * rows
    img = bpy.data.images.new('sheet', W, H, alpha=True)
    px = np.zeros((H, W, 4), dtype=np.float32); px[..., 3] = 1
    for i, p in enumerate(tiles):
        im = bpy.data.images.load(p)
        a = np.array(im.pixels[:], dtype=np.float32).reshape(SIZE, SIZE, 4)
        r, c = rows - 1 - i // cols, i % cols
        px[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE] = a
        bpy.data.images.remove(im); os.remove(p)
    img.pixels = px.ravel()
    img.filepath_raw = path; img.file_format = 'PNG'; img.save()


# ── weights ────────────────────────────────────────────────────────────────
bone_names = [b.name for b in rig.data.bones]
colour = {}
for i, n in enumerate(sorted(bone_names)):
    h = (i * 0.618034) % 1.0
    colour[n] = colorsys.hsv_to_rgb(h, 0.75, 0.95)
attr = mesh.data.color_attributes.new('weights', 'FLOAT_COLOR', 'POINT')
for v in mesh.data.vertices:
    c = [0.0, 0.0, 0.0]; tot = 0
    for g in v.groups:
        n = gnames[g.group]
        if n in colour:
            for k in range(3):
                c[k] += colour[n][k] * g.weight
            tot += g.weight
    attr.data[v.index].color = (*(x / tot for x in c), 1.0) if tot > 0 else (0, 0, 0, 1)
mesh.data.color_attributes.active_color = attr
sh.color_type = 'VERTEX'
tiles = []
for view in ('side', 'front', 'top', 'q34'):
    aim(view); p = os.path.join(OUT, f'.__w_{view}.png'); render(p); tiles.append(p)
compose(tiles, 4, os.path.join(OUT, 'weights.png'))
json.dump({n: [round(x, 3) for x in colour[n]] for n in colour}, open(os.path.join(OUT, 'weight_legend.json'), 'w'))

# ── clips ──────────────────────────────────────────────────────────────────
# a stand-in with no image texture renders by its material colour
sh.color_type = 'TEXTURE' if any(i.size[0] > 0 for i in bpy.data.images) else 'MATERIAL'
for act in actions:
    set_action(act)
    a, b = frame_range(act)
    frames = [round(a + (b - a) * i / (NFR - 1)) for i in range(NFR)] if b > a else [a]
    tiles = []
    for view in ('side', 'front', 'q34'):
        aim(view)
        for f in frames:
            sc.frame_set(f)
            p = os.path.join(OUT, f'.__c_{act.name}_{view}_{f}.png'); render(p); tiles.append(p)
    compose(tiles, len(frames), os.path.join(OUT, f'clip_{act.name}.png'))
# ── full-motion preview ────────────────────────────────────────────────────
if opt.get('video') == '1':
    import subprocess, shutil
    order = [a for n in ('idle', 'idle_flap', 'idle_croak', 'walk', 'walk', 'attack', 'special_1', 'special_2',
                         'special_3', 'death')
             for a in actions if a.name == n]
    seq = os.path.join(OUT, '.__video'); shutil.rmtree(seq, ignore_errors=True); os.makedirs(seq)
    aim('q34')
    k = 0
    for act in order:
        set_action(act)
        a, b = frame_range(act)
        last = b - 1 if act.name.startswith(('idle', 'walk')) else b   # loops: the closing key IS frame a
        for f in range(a, last + 1):
            sc.frame_set(f)
            sc.render.filepath = os.path.join(seq, f'{k:05d}.png'); bpy.ops.render.render(write_still=True); k += 1
        if not act.name.startswith(('idle', 'walk')):  # hold a one-shot's final pose a beat
            for _ in range(20 if act.name == "death" else 8):
                shutil.copy(os.path.join(seq, f'{k - 1:05d}.png'), os.path.join(seq, f'{k:05d}.png')); k += 1
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', os.path.join(seq, '%05d.png'),
                    '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    os.path.join(OUT, 'preview.mp4')], check=True)
    shutil.rmtree(seq, ignore_errors=True)
    print('VIDEO', os.path.join(OUT, 'preview.mp4'), k, 'frames')
print('REVIEW_DONE', OUT)
