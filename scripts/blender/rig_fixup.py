# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""
rig_fixup.py: post-process a Meshy auto-rigged humanoid GLB before the
Mixamo chain (repair_head_weights.py -> retarget_mixamo.py ->
polish_clips.py -> clip_review.py).

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/rig_fixup.py -- \
      --spec examples/anim-authoring/dwarf_axe_shield.json \
      --out output/anim-authoring/dwarf_axe_shield/rig-v1/character.glb

Reads the spec's "rigFix" block. Every step is optional and runs in this order:

  input      Meshy rigged GLB (armature at 0.01 scale, one skinned mesh). Meshy
             also ships a stray unparented 'Icosphere'; unparented meshes are
             dropped.
  weapons    [{"hand": "RightHand", "forearm": "RightForeArm",
               "fistRadius": 0.055, "beyond": 0.0, "exclude": [box, ...]}]
             Meshy skins a held weapon by distance, so a staff crystal ends up
             on the upper arm and an axe head on the forearm, and the prop
             bends when the elbow does. Every vertex past the wrist (along
             elbow->wrist, from `beyond` x height) and farther than
             `fistRadius` x height from that axis is set to 100% hand. The
             fist itself keeps Meshy's weights.
  regions    [{"box": [[x0,y0,z0],[x1,y1,z1]] | "capsule": [[a],[b],r],
               "weights": {"Head": 1.0}, "note": ".."}]: exact rigid weights.
  skirts     [{"box": .., "exclude": [box, ..], "waistZ": z, "hemZ": z,
               "hipsAtWaist": 0.8, "hipsAtHem": 0.3, "ownSide": 0.65}]
             A robe or cloak that Meshy split between the legs tears when
             they stride. In the box, a vertex's leg-chain weight (UpLeg, Leg,
             Foot, Toe of both sides) is moved to Hips (share falling from
             hipsAtWaist at the waist to hipsAtHem at the hem) and to the two
             THIGHS, mostly its own side (ownSide), so the hem swings as a bell
             with the average of the thighs and never follows the shins.
  tail       {"root": [x,y,z], "tip": [x,y,z], "rootRadius": r, "yMin": y,
              "rootDir": [x,y,z] (default tip - root), "bones": 4, "parent": "Hips"}
             Meshy's humanoid rig has no tail. The tail is found by a flood
             fill over the WELDED mesh from the vertex nearest `tip`, never
             entering the root sphere, crossing y < yMin (legs) or the plane
             through the root perpendicular to rootDir (buttocks). Its centre
             line comes from geodesic distance bins; bones tail_1..tail_N run
             root -> tip (parent -> Hips). Weights interpolate linearly between
             bone midpoints and blend into Hips over the first bone.
  shield     {"glb": path, "bone": "LeftForeArm", "size": 0.55,
              "sizeRef": "headTop"|"height", "faceLocal": [0.1,0,-1],
              "downLocal": [1,0,0], "along": 0.5, "gap": 0.004,
              (or "tposeFace": [0.07,0.03,1], "tposeDown": [0.17,0.98,-0.05],
              "tposeAxis": [1,0,0]: the same guard given in the Mixamo T-pose
              frame, for rigs whose forearm roll or rest pose differ, e.g. a
              mounted rider built by rig_mount.py),
              "textureSize": 1024, "back": {...}}
             A separate Meshy shield (front facing -Y, top +Z in its file) is
             scaled so its larger in-plane side is size x the body height,
             oriented in the forearm bone's REST frame (face along faceLocal,
             kite point along downLocal: measured from Mixamo's sword-and-shield
             block/idle, where forearm-local -Z faces the enemy and +X points
             down), pushed out along the face normal until its back clears the
             forearm's own measured surface by `gap`, skinned 100% to `bone`
             and joined into the body as a second material.
             "back": {"style": "planks"|"metal", "colour": [r,g,b],
                      "rim": [r,g,b], "rimWidth": 0.06, "strap": [r,g,b]}
             repaints the back faces (Meshy paints a ghost of the front
             design there) in texture space: back triangles are rasterised in
             UV space and filled from a procedural board/metal image projected
             on the shield plane, with a rim band along the outline, leather
             straps and a grip, then dilated into uncovered texels only.

Writes the skinned GLB (no animations) and <out>.json with per-step counts
and checks (tail vertex count and joints, shield scale/offset, the forearm
clearance, vertices changed). Look at the weight and pose renders
(scripts/blender/meshy_rig_weight_review.py, scripts/blender/rig_pose_check.py) before
trusting it.
"""
import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

LEG_BONES = ['UpLeg', 'Leg', 'Foot', 'ToeBase']


def args_after_dashes():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args(argv)


def in_box(p, box):
    (x0, y0, z0), (x1, y1, z1) = box
    return min(x0, x1) <= p.x <= max(x0, x1) and min(y0, y1) <= p.y <= max(y0, y1) and min(z0, z1) <= p.z <= max(z0, z1)


def in_capsule(p, cap):
    a, b, r = Vector(cap[0]), Vector(cap[1]), cap[2]
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (p - (a + ab * t)).length <= r


def in_shape(p, s):
    if 'box' in s:
        return in_box(p, s['box'])
    if 'capsule' in s:
        return in_capsule(p, s['capsule'])
    return False


class Skin:
    """Per-vertex weight dicts over one mesh; writes back only changed vertices."""

    def __init__(self, obj, arm):
        self.obj, self.arm = obj, arm
        for b in arm.data.bones:
            if b.name not in obj.vertex_groups:
                obj.vertex_groups.new(name=b.name)
        self.names = {vg.index: vg.name for vg in obj.vertex_groups}
        self.w = [{self.names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in obj.data.vertices]
        self.changed = set()

    def set(self, i, d):
        tot = sum(d.values())
        if tot <= 0:
            return
        d = {k: v / tot for k, v in d.items() if v / tot > 1e-4}
        self.w[i] = d
        self.changed.add(i)

    def ensure_group(self, name):
        if name not in self.obj.vertex_groups:
            vg = self.obj.vertex_groups.new(name=name)
            self.names[vg.index] = name

    def write(self, max_inf=4):
        vgs = self.obj.vertex_groups
        for i in self.changed:
            v = self.obj.data.vertices[i]
            for g in list(v.groups):
                vgs[g.group].remove([i])
            items = sorted(self.w[i].items(), key=lambda kv: -kv[1])[:max_inf]
            tot = sum(w for _, w in items)
            for name, w in items:
                vgs[name].add([i], w / tot, 'REPLACE')


def dominant(d):
    return max(d.items(), key=lambda kv: kv[1])[0] if d else None


def welded_graph(obj, P):
    """Adjacency over position-welded vertices (the glTF import splits UV seams)."""
    key = {}
    canon = []
    for i, p in enumerate(P):
        k = (round(p.x, 5), round(p.y, 5), round(p.z, 5))
        canon.append(key.setdefault(k, i))
    adj = {}
    for e in obj.data.edges:
        a, b = canon[e.vertices[0]], canon[e.vertices[1]]
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    members = {}
    for i, c in enumerate(canon):
        members.setdefault(c, []).append(i)
    return canon, adj, members


# ─── weapons ────────────────────────────────────────────────────────────────
def fix_weapons(skin, P, bones, H, cfgs, rep):
    out = []
    for c in cfgs:
        wrist, elbow = bones[c['hand']][0], bones[c['forearm']][0]
        axis = (wrist - elbow).normalized()
        fist_r = c.get('fistRadius', 0.055) * H
        start = c.get('beyond', 0.0) * H
        excl = c.get('exclude', [])
        n = 0
        before = {}
        for i, p in enumerate(P):
            d = p - wrist
            a = d.dot(axis)
            if a < start:
                continue
            if (d - axis * a).length <= fist_r:
                continue
            if any(in_shape(p, e) for e in excl):
                continue
            dom = dominant(skin.w[i])
            before[dom] = before.get(dom, 0) + 1
            skin.set(i, {c['hand']: 1.0})
            n += 1
        out.append({'hand': c['hand'], 'verts': n, 'fistRadius': round(fist_r, 4), 'dominantBefore': before})
    rep['weapons'] = out


def fix_regions(skin, P, cfgs, rep):
    out = []
    for c in cfgs:
        n = 0
        for i, p in enumerate(P):
            if in_shape(p, c) and not any(in_shape(p, e) for e in c.get('exclude', [])):
                skin.set(i, dict(c['weights']))
                n += 1
        out.append({'note': c.get('note', ''), 'verts': n})
    rep['regions'] = out


def fix_skirts(skin, P, cfgs, rep):
    out = []
    for c in cfgs:
        wz, hz = c['waistZ'], c['hemZ']
        hw, hh, own = c.get('hipsAtWaist', 0.8), c.get('hipsAtHem', 0.3), c.get('ownSide', 0.65)
        sideW = c.get('sideWidth', 0.1)
        n = 0
        for i, p in enumerate(P):
            if not in_box(p, c['box']) or any(in_shape(p, e) for e in c.get('exclude', [])):
                continue
            w = dict(skin.w[i])
            legs = {s: sum(w.pop(f'{s}{b}', 0.0) for b in LEG_BONES) for s in ('Left', 'Right')}
            tot = legs['Left'] + legs['Right']
            if tot < 0.02:
                continue
            t = max(0.0, min(1.0, (wz - p.z) / max(wz - hz, 1e-6)))
            hs = hw + (hh - hw) * t
            mine = 'Left' if p.x > 0 else 'Right'
            other = 'Right' if mine == 'Left' else 'Left'
            share = 0.5 + (own - 0.5) * min(1.0, abs(p.x) / sideW)
            w['Hips'] = w.get('Hips', 0.0) + tot * hs
            w[f'{mine}UpLeg'] = w.get(f'{mine}UpLeg', 0.0) + tot * (1 - hs) * share
            w[f'{other}UpLeg'] = w.get(f'{other}UpLeg', 0.0) + tot * (1 - hs) * (1 - share)
            skin.set(i, w)
            n += 1
        out.append({'note': c.get('note', ''), 'verts': n})
    rep['skirts'] = out


# ─── tail ───────────────────────────────────────────────────────────────────
def build_tail(obj, arm, skin, P, H, c, rep):
    canon, adj, members = welded_graph(obj, P)
    root, tip = Vector(c['root']), Vector(c['tip'])
    rr, ymin = c['rootRadius'], c.get('yMin', -1e9)
    cut = Vector(c.get('rootDir', list(tip - root))).normalized()
    seed = canon[min(range(len(P)), key=lambda i: (P[i] - tip).length)]

    def ok(ci):
        # outside the root sphere, clear of the legs (y) and past the root plane
        # (perpendicular to the tail's root direction), so the fill cannot
        # climb over the buttocks or the back
        p = P[ci]
        return (p - root).length > rr and p.y >= ymin and (p - root).dot(cut) >= 0
    # Dijkstra geodesic distance from the tip over the admissible welded set
    dist = {seed: 0.0}
    pq = [(0.0, seed)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in adj.get(u, ()):
            if not ok(v):
                continue
            nd = d + (P[u] - P[v]).length
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    cset = list(dist)
    L = max(dist.values())
    n = int(c.get('bones', 4))
    # joints: root, then centroids of distance bins from the root side to the tip
    joints = [root]
    for k in range(1, n + 1):
        target = L * (1 - k / n)  # geodesic distance from tip
        band = [P[ci] for ci in cset if abs(dist[ci] - target) <= L / (2 * n)]
        if k == n:
            band = [P[ci] for ci in cset if dist[ci] <= L / (2 * n)]
        joints.append(sum(band, Vector()) / len(band))
    # bones
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    inv = arm.matrix_world.inverted()
    eb = arm.data.edit_bones
    names = []
    parent = eb[c.get('parent', 'Hips')]
    for k in range(n):
        name = f'tail_{k + 1}'
        b = eb.new(name)
        b.head = inv @ joints[k]
        b.tail = inv @ joints[k + 1]
        d = (joints[k + 1] - joints[k]).normalized()
        b.align_roll(Vector((0, 0, 1)) if abs(d.z) < 0.8 else Vector((0, 1, 0)))
        b.parent = parent
        b.use_connect = k > 0
        parent = b
        names.append(name)
    bpy.ops.object.mode_set(mode='OBJECT')
    for nm in names:
        skin.ensure_group(nm)
    # weights: projection onto the polyline, linear between bone midpoints
    segs = [(joints[k], joints[k + 1]) for k in range(n)]
    cum = [0.0]
    for a, b in segs:
        cum.append(cum[-1] + (b - a).length)
    mids = [(cum[k] + cum[k + 1]) / 2 for k in range(n)]
    parent_name = c.get('parent', 'Hips')
    tail_verts = [i for ci in cset for i in members[ci]]
    for i in tail_verts:
        p = P[i]
        best = None
        for k, (a, b) in enumerate(segs):
            ab = b - a
            t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
            dd = (p - (a + ab * t)).length
            if best is None or dd < best[0]:
                best = (dd, cum[k] + t * ab.length)
        s = best[1]
        if s <= mids[0]:
            f = s / max(mids[0], 1e-6)
            w = {parent_name: 1 - f, names[0]: f}
        elif s >= mids[-1]:
            w = {names[-1]: 1.0}
        else:
            k = max(j for j in range(n) if mids[j] <= s)
            f = (s - mids[k]) / max(mids[k + 1] - mids[k], 1e-6)
            w = {names[k]: 1 - f, names[k + 1]: f}
        skin.set(i, w)
    rep['tail'] = {'verts': len(tail_verts), 'welded': len(cset), 'geodesicLength': round(L, 4),
                   'joints': [[round(x, 4) for x in j] for j in joints], 'bones': names}


# ─── shield ─────────────────────────────────────────────────────────────────
def base_image(mat):
    if not mat or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            link = node.inputs['Base Color'].links
            if link and link[0].from_node.type == 'TEX_IMAGE':
                return link[0].from_node.image
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE':
            return node.image
    return None


def value_noise(shape, scale, rng):
    """Smooth 2D value noise in [0,1] (bilinear upsample of a random grid)."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def back_pattern(cfg, N=512, seed=7):
    """Procedural back image over the shield's (x, z) bbox: rows = z (top first)."""
    rng = np.random.default_rng(seed)
    style = cfg.get('style', 'planks')
    col = np.array(cfg.get('colour', [0.24, 0.15, 0.09] if style == 'planks' else [0.42, 0.44, 0.47]))
    img = np.ones((N, N, 3)) * col
    X = np.linspace(0, 1, N)[None, :].repeat(N, 0)
    Z = np.linspace(1, 0, N)[:, None].repeat(N, 1)
    if style == 'planks':
        nb = cfg.get('boards', 6)
        board = np.minimum(np.floor(X * nb).astype(int), nb - 1)
        tint = 0.85 + 0.3 * rng.random(nb)
        img *= tint[board][..., None]
        grain = value_noise((N, N), 6, rng)[:, :] * 0.5 + value_noise((N, N), 40, rng) * 0.5
        grain = np.sin((X * nb * 9 + grain * 3.0) * math.pi * 2) * 0.5 + 0.5
        img *= (0.82 + 0.18 * grain)[..., None]
        seam = np.abs((X * nb) - np.round(X * nb)) < 0.03
        img[seam] *= 0.45
    else:
        brushed = value_noise((N, N), 3, rng) * 0.4 + value_noise((N, N), 25, rng) * 0.6
        img *= (0.85 + 0.25 * brushed)[..., None]
        # a dished centre (the inside of the boss) and a ring of rivets
        r = np.hypot(X - 0.5, Z - 0.5)
        img[r < 0.12] *= 0.6
        img[(r > 0.12) & (r < 0.14)] *= 1.25
    strap = np.array(cfg.get('strap', [0.33, 0.19, 0.09]))
    # two straps crossing the forearm line (vertical in the shield's frame) and a grip bar
    for sx in (0.36, 0.64):
        m = (np.abs(X - sx) < 0.045) & (np.abs(Z - 0.5) < 0.26)
        img[m] = strap * (0.9 + 0.2 * value_noise((N, N), 8, rng)[m])[:, None]
        edge = (np.abs(np.abs(X - sx) - 0.045) < 0.006) & (np.abs(Z - 0.5) < 0.26)
        img[edge] = strap * 0.55
        for rz in (0.26, 0.74):
            rv = np.hypot(X - sx, Z - rz) < 0.014
            img[rv] = [0.62, 0.62, 0.6]
    grip = (np.abs(Z - 0.5) < 0.03) & (np.abs(X - 0.5) < 0.2)
    img[grip] = strap * 0.75
    return np.clip(img, 0, 1)


def raster_tris(uv_tris, w, h, fn):
    """Call fn(tri_index, px_x, px_y, bary(n,3)) for texel centres inside each UV triangle."""
    for ti, uv in enumerate(uv_tris):
        pts = np.array(uv) * [w, h]
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        gx, gy = gx.ravel(), gy.ravel()
        (ax, ay), (bx, by), (cx, cy) = pts
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        l3 = 1 - l1 - l2
        m = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
        if m.any():
            fn(ti, gx[m].astype(int), gy[m].astype(int), np.stack([l1[m], l2[m], l3[m]], 1))


def repaint_back(obj, cfg, rep):
    me = obj.data
    me.calc_loop_triangles()
    mat = obj.material_slots[0].material if obj.material_slots else None
    img = base_image(mat)
    if img is None:
        rep['back'] = 'no base image'
        return
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    uvl = me.uv_layers.active.data
    co = np.array([v.co[:] for v in me.vertices])
    xs, zs = co[:, 0], co[:, 2]
    x0, x1, z0, z1 = xs.min(), xs.max(), zs.min(), zs.max()
    N = 512
    pat = back_pattern(cfg, N)
    # outline mask over the (x, z) plane, for the rim band
    mask = np.zeros((N, N), bool)
    xz_tris = []
    uv_tris, back = [], []
    for t in me.loop_triangles:
        vs = [co[i] for i in t.vertices]
        uv_tris.append([uvl[li].uv[:] for li in t.loops])
        xz_tris.append([((v[0] - x0) / (x1 - x0), (v[2] - z0) / (z1 - z0)) for v in vs])
        back.append(t.normal.y > cfg.get('backNormal', 0.35))

    def mark(ti, gx, gy, bary):
        mask[gy, gx] = True
    raster_tris(xz_tris, N, N, mark)
    inner = mask.copy()
    rimpx = max(1, int(cfg.get('rimWidth', 0.06) * N))
    for _ in range(rimpx):
        inner[1:-1, 1:-1] &= inner[:-2, 1:-1] & inner[2:, 1:-1] & inner[1:-1, :-2] & inner[1:-1, 2:]
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    rim = mask & ~inner
    rimcol = np.array(cfg.get('rim', [0.4, 0.41, 0.44]))
    pat[np.flipud(rim)] = rimcol  # pat rows run top (z1) -> bottom; mask rows run z0 -> z1

    cover = np.zeros((h, w), np.int8)  # 1 front, 2 back
    counts = {'front': 0, 'back': 0}

    def paint(ti, gx, gy, bary):
        if back[ti]:
            q = np.array(xz_tris[ti])
            u = bary @ q[:, 0]
            v = bary @ q[:, 1]
            ix = np.clip((u * (N - 1)).astype(int), 0, N - 1)
            iz = np.clip(((1 - v) * (N - 1)).astype(int), 0, N - 1)
            px[gy, gx, :3] = pat[iz, ix]
            cover[gy, gx] = 2
            counts['back'] += len(gx)
        else:
            cover[gy, gx] = np.where(cover[gy, gx] == 2, 2, 1)
            counts['front'] += len(gx)
    raster_tris(uv_tris, w, h, paint)
    # dilate back texels into texels no triangle covers (never over front texels)
    grown = cover == 2
    for _ in range(3):
        nb = np.zeros_like(grown)
        src = np.zeros((h, w, 3), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(grown, dy, 0), dx, 1)
            cand = sh & ~grown & (cover == 0) & ~nb
            src[cand] = np.roll(np.roll(px[..., :3], dy, 0), dx, 1)[cand]
            nb |= cand
        px[nb, :3] = src[nb]
        grown |= nb
    img.pixels.foreach_set(px.ravel())
    img.pack()
    rep['back'] = {'style': cfg.get('style', 'planks'), 'backTexels': counts['back'], 'frontTexels': counts['front'],
                   'backTris': int(sum(back)), 'tris': len(back)}


def attach_shield(arm, body, skin, P, bones, H, head_top, c, rep):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(Path(c['glb']).resolve()))
    new = [o for o in bpy.data.objects if o not in before]
    shields = [o for o in new if o.type == 'MESH']
    for o in new:
        if o.type != 'MESH':
            bpy.data.objects.remove(o)
    if len(shields) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in shields:
            o.select_set(True)
        bpy.context.view_layer.objects.active = shields[0]
        bpy.ops.object.join()
    sh = shields[0]
    sh.parent = None
    sh.data.transform(sh.matrix_world)  # bake the import transform: shield space = file space (front -Y, up +Z)
    sh.matrix_world = Matrix.Identity(4)
    co = [v.co.copy() for v in sh.data.vertices]
    lo = Vector([min(v[i] for v in co) for i in range(3)])
    hi = Vector([max(v[i] for v in co) for i in range(3)])
    if c.get('back'):
        repaint_back(sh, c['back'], rep.setdefault('shield', {}))
    # origin: bbox centre in x/z, back-most point in y
    origin = Vector(((lo.x + hi.x) / 2, hi.y, (lo.z + hi.z) / 2))
    sh.data.transform(Matrix.Translation(-origin))
    ext = max(hi.x - lo.x, hi.z - lo.z)
    ref = head_top if c.get('sizeRef', 'headTop') == 'headTop' else H
    scale = c['size'] * ref / ext
    # forearm rest frame (world)
    bone = arm.data.bones[c['bone']]
    if 'tposeFace' in c:
        # Orientation given in the Mixamo T-pose frame (the retarget swings each
        # limb to Mixamo's rest direction before applying deltas): undo the
        # minimal rotation that takes this forearm's rest direction to the
        # T-pose one, so a rider whose arm hangs down gets the same guard.
        e = bones[c['bone']][0]
        wr = bones[bone.children[0].name][0] if bone.children else bones[c['bone']][1]
        A = (wr - e).normalized().rotation_difference(Vector(c.get('tposeAxis', [1, 0, 0])).normalized())
        F = (A.inverted() @ Vector(c['tposeFace'])).normalized()
        D = A.inverted() @ Vector(c.get('tposeDown', [0.17, 0.98, -0.05]))
    else:
        R = (arm.matrix_world @ bone.matrix_local).to_3x3().normalized()
        F = (R @ Vector(c.get('faceLocal', [0.1, 0, -1]))).normalized()
        D = R @ Vector(c.get('downLocal', [1, 0, 0]))
    D = (D - F * D.dot(F)).normalized()
    sy, sz = -F, -D
    sx = sy.cross(sz).normalized()
    M = Matrix((sx, sy, sz)).transposed()
    # bone tails from the glTF importer are guesses (Meshy's come out metres
    # long), so the forearm ends at its child's head: the wrist
    elbow = bones[c['bone']][0]
    wrist = bones[bone.children[0].name][0] if bone.children else bones[c['bone']][1]
    axis = (wrist - elbow)
    # forearm surface toward the shield: 90th percentile of the offset along F
    offs = []
    for i, p in enumerate(P):
        if dominant(skin.w[i]) != c['bone']:
            continue
        d = p - elbow
        a = d.dot(axis) / axis.length_squared
        if 0.15 < a < 0.85:
            perp = d - axis * a
            offs.append(perp.dot(F))
    offs.sort()
    armR = offs[int(0.9 * (len(offs) - 1))] if offs else 0.04 * H
    centre = elbow + axis * c.get('along', 0.5) + F * (armR + c.get('gap', 0.004))
    T = Matrix.Translation(centre) @ M.to_4x4() @ Matrix.Diagonal((scale, scale, scale, 1))
    sh.data.transform(T)
    img = base_image(sh.material_slots[0].material) if sh.material_slots else None
    if img is not None and c.get('textureSize'):
        s = int(c['textureSize'])
        if img.size[0] > s:
            img.scale(s, s)
            img.pack()
    # clearance: body vertices (not on this arm chain) inside the shield slab
    sv = [v.co.copy() for v in sh.data.vertices]
    shield_w = scale * ext
    inside = 0
    arm_chain = {c['bone'], bone.parent.name if bone.parent else '', bone.children[0].name if bone.children else ''}
    for i, p in enumerate(P):
        if dominant(skin.w[i]) in arm_chain:
            continue
        d = p - centre
        if -0.002 < d.dot(F) < (hi.y - lo.y) * scale and abs(d.dot(sx)) < shield_w / 2 and abs(d.dot(sz)) < shield_w / 2:
            inside += 1
    # skin it: parent to the armature, one group at 100%
    sh.parent = arm
    sh.matrix_parent_inverse = arm.matrix_world.inverted()
    vg = sh.vertex_groups.new(name=c['bone'])
    vg.add(list(range(len(sh.data.vertices))), 1.0, 'REPLACE')
    mod = sh.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm
    rep.setdefault('shield', {}).update({
        'glb': c['glb'], 'bone': c['bone'], 'scale': round(scale, 5), 'sizeM': round(shield_w, 4),
        'forearmSurface': round(armR, 4), 'centre': [round(x, 4) for x in centre],
        'face': [round(x, 3) for x in F], 'down': [round(x, 3) for x in D],
        'bodyVertsInsideSlab': inside, 'verts': len(sv)})
    # join into the body (second material)
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    sh.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()


def main():
    args = args_after_dashes()
    spec = json.loads(Path(args.spec).read_text())
    F = spec['rigFix']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(Path(F['input']).resolve()))
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    stray = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.parent is None]
    rep = {'input': F['input'], 'strayDropped': [o.name for o in stray]}
    for o in stray:
        bpy.data.objects.remove(o)
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    assert len(meshes) == 1, [o.name for o in meshes]
    body = meshes[0]
    W = body.matrix_world
    P = [W @ v.co for v in body.data.vertices]
    zs = [p.z for p in P]
    H = max(zs) - min(zs)
    bones = {b.name: (arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local) for b in arm.data.bones}
    skin = Skin(body, arm)
    headset = {'Head', 'head_end', 'headfront', 'neck'}
    head_top = max((p.z for i, p in enumerate(P) if dominant(skin.w[i]) in headset), default=max(zs)) - min(zs)
    rep.update({'height': round(H, 4), 'headTop': round(head_top, 4), 'verts': len(P)})
    if F.get('weapons'):
        fix_weapons(skin, P, bones, H, F['weapons'], rep)
    if F.get('regions'):
        fix_regions(skin, P, F['regions'], rep)
    if F.get('skirts'):
        fix_skirts(skin, P, F['skirts'], rep)
    if F.get('tail'):
        build_tail(body, arm, skin, P, H, F['tail'], rep)
    rep['changedVerts'] = len(skin.changed)
    skin.write()
    if F.get('shield'):
        attach_shield(arm, body, skin, P, bones, H, head_top, F['shield'], rep)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True,
                              export_animations=False, export_skins=True, export_all_influences=False,
                              export_yup=True, export_apply=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    out.with_suffix('.json').write_text(json.dumps(rep, indent=2) + '\n')
    print('RIGFIX', json.dumps({k: rep[k] for k in rep if k not in ('input',)}))


if __name__ == '__main__':
    main()
