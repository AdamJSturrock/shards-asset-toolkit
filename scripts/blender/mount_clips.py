# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""
mount_clips.py: author a mounted unit's clip set on a rig built by
scripts/blender/rig_mount.py: a procedural mount (critter-quadruped motion on
the `horse_` bones) under a Mixamo rider (retargeted onto the rider's Meshy
bones by scripts/blender/retarget_mixamo.py), one action per game clip, so the
runtime's single mixer plays horse and rider together.

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/mount_clips.py -- \
      --spec examples/anim-authoring/mounted_knight.json \
      --out output/anim-authoring/mounted_knight/v1/character.glb

Spec "mountClips":
  rig          skinned GLB from rig_mount.py
  candidates   retarget output on that rig (rider actions)
  riderLock    {"legs": 0, "Hips": 0.15, "Spine02": 0.7, ...}: fraction of each
               rider bone's local rotation kept (legs 0 = locked in the
               stirrups); the rider's Hips translation is always dropped
  mount        {"stride": 20, "flex": 45, "nod": 6, "tailSway": 10,
               "gallopStride": 34, "bodyHalfWidth": 0.3}
  clips        {name: {"frames": n, "loop": bool, "horse": "idle" | "look" |
                "walk" | "gallop" | "stamp" | "rear" | "death",
                "horseArgs": {...}, "rider": {"source": act, "range": [a, b],
                "fit": true | "hold": frame} | null, "markers": {"hit": frac},
                "groundClamp": bool}}
               rider.fit resamples the source range onto the clip's frames
               (a closed Mixamo loop stays closed); rider.hold freezes one
               source frame; markers are fractions of the clip.

Writes the GLB (all actions), .blend and <out>.clips.json (seconds, loop,
events in seconds, ground lift) like polish_clips.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polish_clips import (FPS, ground_clamp, ignored_vertices, import_glb,  # noqa: E402
                                  lowest_point, sample, write_action)

X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
RIDER_LEGS = [f'{s}{b}' for s in ('Left', 'Right') for b in ('UpLeg', 'Leg', 'Foot', 'ToeBase')]


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def bump(t, a, b, ramp=0.25):
    if t <= a or t >= b:
        return 0.0
    u = (t - a) / (b - a)
    return smooth(u / ramp) if u < ramp else (smooth((1 - u) / ramp) if u > 1 - ramp else 1.0)


class Pose:
    def __init__(self, rest):
        self.rest = rest
        self.q = {}
        self.root_loc = Vector()

    def rot(self, bn, axis, deg):
        if bn in self.rest and deg:
            r = self.rest[bn]
            self.q[bn] = (r.inverted() @ Quaternion(axis, math.radians(deg)) @ r) @ self.q.get(bn, Quaternion())


H = 'horse_'
QLEGS = [(H + 'thigh.L', H + 'shin.L', H + 'foot.L', 0.0, 'hind'), (H + 'upperarm.L', H + 'forearm.L', H + 'hand.L', 0.25, 'front'),
         (H + 'thigh.R', H + 'shin.R', H + 'foot.R', 0.5, 'hind'), (H + 'upperarm.R', H + 'forearm.R', H + 'hand.R', 0.75, 'front')]


def tail_bones(rest):
    out, i = [], 1
    while f'{H}tail_{i}' in rest:
        out.append(f'{H}tail_{i}')
        i += 1
    return out


def legs_cycle(p, t, A, F, phases, duty):
    """World-pitch leg chains (scripts/critter quad_walk): stance sweeps
    back straight, swing folds the lower leg and brings it forward."""
    for (up, lo, ft, _, kind), ph in zip(QLEGS, phases):
        c = (t + ph) % 1.0
        if c < duty:
            s = c / duty
            w_up = -A + 2 * A * s
            w_lo, w_ft = w_up, 0.0
        else:
            s = (c - duty) / (1 - duty)
            w_up = A - 2 * A * smooth(s)
            lift = math.sin(math.pi * s)
            if kind == 'front':
                w_lo = w_up + 1.35 * F * lift
                w_ft = 0.5 * w_lo
            else:
                w_lo = w_up + 0.8 * F * lift
                w_ft = w_lo - 1.1 * F * lift
        p.rot(up, X, w_up)
        p.rot(lo, X, w_lo - w_up)
        p.rot(ft, X, w_ft - w_lo)


def horse_walk(p, t, M, a):
    legs_cycle(p, t, M.get('stride', 20), M.get('flex', 45), [0.0, 0.25, 0.5, 0.75], 0.62)
    w = 2 * math.pi * t
    p.rot(H + 'chest', X, 1.5 * math.sin(2 * w))
    roll = 2.5 * math.sin(w)
    p.rot(H + 'hips', Y, roll)
    nod = M.get('nod', 6)
    p.rot(H + 'neck', X, nod * math.sin(2 * w + 0.6))
    p.rot(H + 'head', X, -0.5 * nod * math.sin(2 * w + 0.6))
    for i, bn in enumerate(tail_bones(p.rest)):
        p.rot(bn, Z, M.get('tailSway', 10) * (0.6 + 0.25 * i) * math.sin(w - 0.7 * (i + 1)))
    p.root_loc = Vector((0, 0, 0.012 * (1 + math.cos(2 * w))))


def horse_gallop(p, t, M, a):
    """Rotary gallop: hinds together-ish, then fronts; the body rocks and bounds."""
    cyc = a.get('cycles', 2)
    tt = (t * cyc) % 1.0
    legs_cycle(p, tt, M.get('gallopStride', 34), M.get('flex', 45) * 1.3, [0.0, 0.52, 0.1, 0.62], 0.42)
    w = 2 * math.pi * tt
    p.rot(H + 'hips', X, 5 * math.sin(w))
    p.rot(H + 'chest', X, -3 * math.sin(w))
    p.rot(H + 'neck', X, 8 + 7 * math.sin(w + 1.2))
    p.rot(H + 'head', X, -4 * math.sin(w + 1.2))
    for i, bn in enumerate(tail_bones(p.rest)):
        p.rot(bn, X, -25 - 5 * i)
        p.rot(bn, Z, 6 * math.sin(w - 0.6 * (i + 1)))
    p.root_loc = Vector((0, 0, 0.06 * max(0.0, math.sin(w))))


def horse_idle(p, t, M, a):
    style = a.get('style', 'idle')
    p.rot(H + 'chest', X, 1.2 * math.sin(2 * math.pi * 2 * t))
    if style == 'look':
        look = 22 * bump(t, 0.05, 0.45) - 18 * bump(t, 0.55, 0.95)
        p.rot(H + 'neck', Z, look * 0.6)
        p.rot(H + 'head', Z, look * 0.4)
        p.rot(H + 'neck', X, -6 * bump(t, 0.05, 0.95))
    else:
        p.rot(H + 'neck', X, 3 * math.sin(2 * math.pi * t) + 5 * bump(t, 0.4, 0.62))
        p.rot(H + 'head', X, 6 * bump(t, 0.4, 0.62) * math.sin(2 * math.pi * 6 * t))  # a head toss
        # rest a hind hoof, then shift back
        rest = bump(t, 0.15, 0.55)
        p.rot(H + 'shin.R', X, 10 * rest)
        p.rot(H + 'foot.R', X, -14 * rest)
    for i, bn in enumerate(tail_bones(p.rest)):
        p.rot(bn, Z, (6 + 4 * i) * math.sin(2 * math.pi * 2 * t - 0.7 * i) * (0.4 + bump(t, 0.6, 0.9)))


def horse_stamp(p, t, M, a):
    """Paw the ground with the left fore, weight back, head up: a light attack or a cue."""
    amp = a.get('amp', 1.0)
    paw = bump(t, 0.1, 0.8, 0.3) * amp
    strike = math.sin(math.pi * 2 * smooth((t - 0.15) / 0.6)) * bump(t, 0.15, 0.75)
    p.rot(H + 'upperarm.L', X, -35 * paw - 10 * strike)
    p.rot(H + 'forearm.L', X, 70 * paw)
    p.rot(H + 'hand.L', X, -20 * paw)
    p.rot(H + 'hips', X, -4 * paw)
    p.rot(H + 'thigh.L', X, 4 * paw)
    p.rot(H + 'thigh.R', X, 4 * paw)
    p.rot(H + 'neck', X, -10 * paw)
    p.rot(H + 'head', X, 6 * paw)


def horse_rear(p, t, M, a):
    """Rear up about the rump: hips pitch the front up, the hind legs are
    counter-rotated to stay planted, the forelegs fold and strike."""
    up = a.get('deg', 22) * bump(t, 0.05, 0.9, 0.35)
    p.rot(H + 'hips', X, -up)
    for s in ('L', 'R'):
        p.rot(f'{H}thigh.{s}', X, up)
    fold = bump(t, 0.05, 0.9, 0.3)
    strike = math.sin(math.pi * 3 * smooth((t - 0.2) / 0.6)) * bump(t, 0.2, 0.8)
    for s, off in (('L', 0.0), ('R', 0.6)):
        p.rot(f'{H}upperarm.{s}', X, -40 * fold - 18 * strike * (1 - off))
        p.rot(f'{H}forearm.{s}', X, 85 * fold)
        p.rot(f'{H}hand.{s}', X, -25 * fold)
    p.rot(H + 'neck', X, -14 * fold)
    p.rot(H + 'head', X, 10 * fold)
    for i, bn in enumerate(tail_bones(p.rest)):
        p.rot(bn, X, 15 * fold)


def horse_death(p, t, M, a):
    """Front knees buckle, then the horse keels over onto its left side."""
    kneel = smooth((t - 0.05) / 0.3)
    roll = 88 * smooth((t - 0.3) / 0.45) + 4 * math.sin(math.pi * smooth((t - 0.75) / 0.2))
    for s in ('L', 'R'):
        p.rot(f'{H}upperarm.{s}', X, -15 * kneel)
        p.rot(f'{H}forearm.{s}', X, 60 * kneel * (1 - 0.6 * smooth((t - 0.5) / 0.3)))
    p.rot(H + 'root', Y, roll)
    hw = M.get('bodyHalfWidth', 0.3)
    drop = 0.18 * kneel * (1 - smooth((t - 0.35) / 0.3))
    p.root_loc = Vector((0, 0, hw * math.sin(math.radians(min(roll, 90))) * 0.9 - drop))
    stiff = smooth((t - 0.45) / 0.4)
    p.rot(H + 'neck', X, -12 * bump(t, 0.0, 0.3, 0.4) + 25 * stiff)
    p.rot(H + 'head', X, 12 * stiff)
    for up, lo, ft, _, kind in QLEGS:
        p.rot(up, X, (-15 if kind == 'front' else 15) * stiff)
    for bn in tail_bones(p.rest):
        p.rot(bn, X, -15 * stiff)


HORSE = {'idle': horse_idle, 'look': lambda p, t, M, a: horse_idle(p, t, M, {**a, 'style': 'look'}),
         'walk': horse_walk, 'gallop': horse_gallop, 'stamp': horse_stamp, 'rear': horse_rear, 'death': horse_death}


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    S = spec['mountClips']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS
    rig, rig_objs, stray = import_glb(S['rig'])
    for a in stray:
        bpy.data.actions.remove(a)
    meshes = [o for o in rig_objs if o.type == 'MESH']
    rig.animation_data_create()
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    rest = {b.name: (rig.matrix_world.to_3x3().normalized() @ b.matrix_local.to_3x3()).to_quaternion() for b in rig.data.bones}
    ignore = ignored_vertices(meshes, set(S.get('clampIgnoreBones', [])))
    floor = lowest_point(meshes, ignore)
    src_rig, src_objs, acts = import_glb(S['candidates'])
    src_rig.animation_data_create()
    by = {a.name: a for a in acts}
    lock = S.get('riderLock', {})
    leg_keep = float(lock.get('legs', 0.0))
    keep = {**{b: leg_keep for b in RIDER_LEGS}, **{k: v for k, v in lock.items() if k != 'legs'}}
    M = S.get('mount', {})
    table = {}
    for name, c in S['clips'].items():
        n = int(c['frames'])
        loop = bool(c.get('loop', False))
        count = n + 1 if not loop else n + 1
        rc = c.get('rider')
        if rc:
            act = by[rc['source']]
            lo, hi = rc.get('range', [int(act.frame_range[0]), int(act.frame_range[1])])
            if 'hold' in rc:
                frames = [rc['hold']] * count
            else:
                frames = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
            poses = sample(src_rig, act, frames)
        else:
            poses = [{pb.name: (Vector(), Quaternion()) for pb in src_rig.pose.bones} for _ in range(count)]
        fn = HORSE[c.get('horse', 'idle')]
        hargs = c.get('horseArgs', {})
        for i, pose in enumerate(poses):
            t = i / n
            if loop and i == n:
                t = 0.0
            p = Pose(rest)
            fn(p, t, M, hargs)
            for bn in list(pose):
                loc, q = pose[bn]
                if bn.startswith(H):
                    pose[bn] = (p.root_loc if bn == H + 'root' else Vector(), p.q.get(bn, Quaternion()))
                    continue
                if bn == 'Hips':
                    loc = Vector()
                if bn in keep:
                    q = q.copy()
                    if q.w < 0:
                        q.negate()
                    q = Quaternion().slerp(q, float(keep[bn]))
                pose[bn] = (loc, q)
        # root translation is written for parent-less bones only: horse_root.
        # write_action takes the root location in the root bone's local frame.
        root = rig.data.bones[H + 'root']
        to_local = root.matrix_local.to_3x3().inverted()
        for pose in poses:
            loc, q = pose[H + 'root']
            pose[H + 'root'] = (to_local @ loc, q)
        markers = {k: v * n for k, v in c.get('markers', {}).items()}
        act = write_action(rig, name, poses, markers, loop)
        lift = round(ground_clamp(rig, meshes, act, floor, ignore=ignore), 4) if c.get('groundClamp') else None
        table[name] = {'horse': c.get('horse'), 'rider': rc, 'loop': loop, 'seconds': round(n / FPS, 3),
                       'groundLiftMax': lift, 'events': {k: round(v / FPS, 3) for k, v in markers.items()}}
    for o in src_objs:
        bpy.data.objects.remove(o)
    for a in acts:
        bpy.data.actions.remove(a)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for pb in rig.pose.bones:
        pb.matrix_basis.identity()
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    bpy.ops.object.select_all(action='DESELECT')
    for o in rig_objs:
        if o.name in bpy.data.objects:
            o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True,
                              export_animation_mode='ACTIONS', export_anim_slide_to_zero=True,
                              export_force_sampling=True)
    Path(str(out).replace('.glb', '.clips.json')).write_text(json.dumps({'spec': args.spec, 'fps': FPS, 'clips': table}, indent=2) + '\n')
    print('MOUNTCLIPS', json.dumps({k: {'seconds': v['seconds'], 'lift': v['groundLiftMax']} for k, v in table.items()}))


if __name__ == '__main__':
    main()
