# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Turn retargeted candidate clips into a unit's game clip set, from a JSON spec.

Per output clip the spec picks a source action and applies, in order:
  range      [first, last] source frames to keep
  yawKeep    0..1: fraction of the root's yaw swing kept (RTS units must keep
             facing their target; Mixamo swings spin the hips 90-170 degrees)
  yawRef     "first" (default): the swing is measured from the clip's first
             frame; "rest": from the rig's rest facing, which also turns a
             side-on fighting stance (Mixamo's weapon idles stand ~50 degrees
             off) back toward the unit's facing
  loop       close the seam: already-closed seams (<1 deg) are left alone; a real
             gap is spread evenly over the clip (never a tail blend, which hitches)
  speed      playback-rate multiplier baked into the keys (resampled)
  groundClamp  lift the root on frames where the real skinned mesh dips below
             the rest-pose floor (sleep/sit/death on non-human proportions).
             Spec-level "clampIgnoreBones": ["RightHand"] leaves out vertices
             dominated by those bones (a rigid weapon must not prop a corpse up)
  markers    {"hit": 21, ...} in SOURCE frames; stored as action pose markers
             and exported to <out>.clips.json in seconds from the clip start
  twoHandProp {"hand": "RightHand", "other": "LeftHand", "aim": 1.0}: a prop
             fused to both hands (a musket held across the chest). Spec-level,
             overridable per clip (null disables). The prop and both hands move
             as one rigid unit driven by `hand`: `hand` is re-aimed (by `aim`)
             so the unit's hand-to-hand axis follows the retargeted hands'
             axis (Mixamo's rifle runs from the right hand to the left), then
             `other` is solved by two-bone IK onto its rest place on the prop.
             The aim is reduced per frame, down to 0, until `other`'s place
             is reachable. "slide": true then lets `other` slide along the prop toward `hand`
             when its rest place is out of reach (a claw sliding down the
             barrel), so it stays on the prop. The worst remaining wrist gap
             is reported.
  uprightProp {"hand": "LeftHand", "axis": [[x,y,z], [x,y,z]], "maxTilt": 35}:
             a staff held in one hand. `axis` is the prop's foot and head in
             rest space; the hand is turned (wrist only) whenever the prop
             would tilt more than maxTilt degrees from vertical, so a
             Mixamo cast never swings the staff orb-down. Spec-level default,
             overridable per clip (null disables).
  recoil     {"at": 6, "hand": "RightHand", "axis": [[stock], [muzzle]],
              "bones": {"Spine": 6, "RightArm": 14, "Head": 4}, "settle": 0.25}:
             a procedural gun kick at source frame `at`. Each bone turns by
             its degrees about (prop direction x up), so the muzzle climbs
             and the body rocks back, peaking in one frame and settling over
             `settle` seconds. Applied after twoHandProp, whose off hand is
             then re-solved onto the prop (aim 0), so the grip holds.
  secondary  {"bones": ["tail_1", ...], "swayDeg": 8, "grow": 0.35, "period": 1.6,
              "lag": 0.7, "decay": false}: procedural secondary motion for bones the
             source rig lacks (a tail on a Mixamo clip). Each bone swings about
             world up (in its own rest frame) by swayDeg x (1 + grow x index),
             a sine phase-lagged by `lag` radians per bone down the chain. Loops
             get a whole number of cycles (~period seconds each), so the seam
             stays closed; "decay" fades it out over a one-shot (a death).
             "level": 0.7 makes the first bone cancel that fraction of its
             parent's pitch and roll (yaw kept), so the tail trails behind
             under gravity instead of swinging up with a forward lunge or fall.
             The reported secondaryPeakDeg is the chain's summed amplitude.
             Spec-level default, merged with a per-clip object (null disables).
  boneKeep   {"Head": 0.2, "neck": 0}: fraction of each named bone's own local
             rotation kept (0 = locked to its parent). Spec-level default,
             overridable per clip. For heads fused into a carapace, where a
             Mixamo head turn would swivel the whole shell.

Clip names are the runtime's canonical words (idle, idle_2, walk, attack,
gather, build, sleep, sitting, death) so the Shards of Stone runtime and the VAT baker match them.
Never a bare `sit`: the runtime substring-matches aliases, and 'sit' is inside
'transition' and 'situps'.
`extra` copies actions verbatim from another GLB on the same skeleton (e.g. an
existing, already-approved death clip).

    blender -b --factory-startup --python-exit-code 1 \
      --python scripts/blender/polish_clips.py -- \
      --spec examples/anim-authoring/goblin_worker.json \
      --out output/anim-authoring/goblin_worker/v1/animations.glb

Spec shape:
{
  "rig": "<skinned GLB, head weights repaired>",
  "candidates": "<GLB from retarget_mixamo.py>" or [several, searched in order],
  "extra": {"from": "<GLB>", "actions": {"death": "death"}},
  "clips": {"attack": {"source": "...", "yawKeep": 0.3, "markers": {"hit": 21}}, ...}
}
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

FPS = 30


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args(sys.argv[sys.argv.index('--') + 1:])


def import_glb(path):
    before = set(bpy.data.objects)
    acts_before = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(Path(path).resolve()))
    new = [o for o in bpy.data.objects if o not in before]
    for o in list(new):
        if o.type == 'MESH' and not any(m.type == 'ARMATURE' for m in o.modifiers):
            bpy.data.objects.remove(o)
            new.remove(o)
    rig = next(o for o in new if o.type == 'ARMATURE')
    return rig, new, [a for a in bpy.data.actions if a not in acts_before]


def sample(rig, action, frames):
    """Per frame: {bone: (location, quaternion)} of pose-bone basis transforms."""
    rig.animation_data.action = action
    if action.slots:
        rig.animation_data.action_slot = action.slots[0]
    out = []
    for f in frames:
        bpy.context.scene.frame_set(int(f), subframe=f - int(f))
        out.append({pb.name: (pb.location.copy(), pb.rotation_quaternion.copy() if pb.rotation_mode == 'QUATERNION'
                              else pb.matrix_basis.to_quaternion()) for pb in rig.pose.bones})
    return out


def world_yaw_limit(rig, poses, keep, ref='first'):
    """Scale the root bone's world yaw deviation from its first-frame (or rest) yaw by `keep`."""
    root = next(b for b in rig.data.bones if b.parent is None).name
    rest = rig.matrix_world.to_quaternion() @ rig.data.bones[root].matrix_local.to_quaternion()
    # Swing-twist: the twist of (world_i * world_0^-1) about world up is the yaw
    # swing. Bone-axis projections are unreliable (Meshy's Hips bone points
    # sideways-down), so never read yaw off a bone axis.
    first = rest.copy() if ref == 'rest' else None
    for pose in poses:
        loc, q = pose[root]
        world = rest @ q
        if first is None:
            first = world.copy()
        d = world @ first.inverted()
        yaw = 2 * math.atan2(d.z, d.w)
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        correction = Quaternion((0, 0, 1), -yaw * (1 - keep))
        pose[root] = (loc, (rest.inverted() @ correction @ world).normalized())


def seam_gap(poses):
    """Largest per-bone rotation (degrees) between the last and first pose."""
    worst = 0.0
    for bone, (_, q) in poses[-1].items():
        d = math.degrees(q.rotation_difference(poses[0][bone][1]).angle)
        worst = max(worst, min(d, 360 - d))
    return worst


def close_loop(poses, tolerance_deg=1.0):
    """Close a loop seam without slowing the motion into it.

    Mixamo loops already end on their first pose; blending the tail toward the
    start pose then decelerates the last frames and leaves a visible hitch every
    cycle (a goblin v1 walk: last step 4.7 deg/frame against a 13.8 median).
    So: leave closed seams alone, and spread any real gap evenly over the whole
    clip, so every frame absorbs the same small correction."""
    gap = seam_gap(poses)
    if gap <= tolerance_deg:
        return gap
    n = len(poses)
    first, last = poses[0], poses[-1]
    for i, pose in enumerate(poses):
        t = i / (n - 1)
        fixed = {}
        for bone, (loc, q) in pose.items():
            fl, fq = first[bone]
            ll, lq = last[bone]
            fq = fq.copy()
            fq.make_compatible(lq)
            correction = Quaternion().slerp(fq @ lq.inverted(), t)
            fixed[bone] = (loc + (fl - ll) * t, (correction @ q).normalized())
        poses[i] = fixed
    return gap


def two_hand_prop(rig, poses, cfg):
    """Solve a two-hand prop per frame (see the module doc). Returns the worst wrist gap."""
    hand, other = cfg['hand'], cfg['other']
    aim = float(cfg.get('aim', 1.0))
    side = other[:-4]  # 'Left' from 'LeftHand'
    upper, fore = f'{side}Arm', f'{side}ForeArm'
    bones = rig.data.bones
    pb = rig.pose.bones
    rig.animation_data.action = None
    rest_h, rest_o = bones[hand].matrix_local.copy(), bones[other].matrix_local.copy()
    rel = rest_h.inverted() @ rest_o  # other hand in the driving hand's frame
    v_rest = rest_o.translation - rest_h.translation
    worst = 0.0

    def apply(pose):
        for name, (loc, q) in pose.items():
            pb[name].rotation_mode = 'QUATERNION'
            pb[name].rotation_quaternion = q
            pb[name].location = loc if pb[name].parent is None else Vector()
        bpy.context.view_layer.update()

    def set_world_rot(name, rot):
        m = pb[name].matrix.copy()
        loc = m.translation.copy()
        pb[name].matrix = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
        bpy.context.view_layer.update()

    for pose in poses:
        apply(pose)
        mh = pb[hand].matrix.copy()
        d = pb[other].head - pb[hand].head
        v_cur = mh.to_quaternion() @ rest_h.to_quaternion().inverted() @ v_rest
        s, e, w = pb[upper].head.copy(), pb[fore].head.copy(), pb[other].head.copy()
        a, b = (e - s).length, (w - e).length
        base_q = mh.to_quaternion()
        if aim > 0 and d.length > 1e-6:
            full = v_cur.rotation_difference(d)
            # Largest aim (<= cfg aim) whose off-hand place is reachable: a short
            # arm cannot follow Mixamo's rifle all the way to a shoulder aim.
            for k in range(11):
                a_try = aim * (1 - k / 10)
                q_try = Quaternion().slerp(full, a_try) @ base_q
                place = (Matrix.Translation(mh.translation) @ q_try.to_matrix().to_4x4() @ rel).translation
                if (place - s).length <= 0.995 * (a + b):
                    break
            set_world_rot(hand, q_try)
            mh = pb[hand].matrix.copy()
        goal = mh @ rel
        target = goal.translation
        if cfg.get('slide') and (target - s).length > 0.98 * (a + b):
            base = mh.translation
            for i in range(1, 61):
                t = 1.0 - 0.7 * i / 60
                cand = base + (target - base) * t
                if (cand - s).length <= 0.98 * (a + b):
                    goal = Matrix.Translation(cand) @ goal.to_quaternion().to_matrix().to_4x4()
                    target = cand
                    break
        dist = max(abs(a - b) + 1e-5, min(a + b - 1e-5, (target - s).length))
        n = (target - s).normalized()
        x = (a * a - b * b + dist * dist) / (2 * dist)
        h = max(0.0, a * a - x * x) ** 0.5
        pole = (e - s) - n * (e - s).dot(n)
        pole = pole.normalized() if pole.length > 1e-6 else Vector((0, 0, -1))
        e2 = s + n * x + pole * h
        set_world_rot(upper, (e - s).rotation_difference(e2 - s) @ pb[upper].matrix.to_quaternion())
        e_now, w_now = pb[fore].head.copy(), pb[other].head.copy()
        set_world_rot(fore, (w_now - e_now).rotation_difference(target - e_now) @ pb[fore].matrix.to_quaternion())
        set_world_rot(other, goal.to_quaternion())
        worst = max(worst, (pb[other].head - target).length)
        for name in (hand, upper, fore, other):
            pose[name] = (pose[name][0], pb[name].rotation_quaternion.copy())
    for p in pb:
        p.matrix_basis.identity()
    return worst


def upright_prop(rig, poses, cfg):
    """Limit a one-hand prop's tilt from vertical (see the module doc). Returns the worst tilt before the fix."""
    hand = cfg['hand']
    max_tilt = math.radians(float(cfg.get('maxTilt', 35)))
    foot, head = Vector(cfg['axis'][0]), Vector(cfg['axis'][1])
    pb = rig.pose.bones
    rig.animation_data.action = None
    rest_q = rig.data.bones[hand].matrix_local.to_quaternion()
    local_axis = rest_q.inverted() @ (head - foot).normalized()
    up = Vector((0, 0, 1))
    worst = 0.0
    for pose in poses:
        for name, (loc, q) in pose.items():
            pb[name].rotation_mode = 'QUATERNION'
            pb[name].rotation_quaternion = q
            pb[name].location = loc if pb[name].parent is None else Vector()
        bpy.context.view_layer.update()
        m = pb[hand].matrix.copy()
        d = m.to_quaternion() @ local_axis
        tilt = d.angle(up)
        worst = max(worst, tilt)
        if tilt > max_tilt:
            axis = d.cross(up)
            if axis.length < 1e-6:
                axis = Vector((1, 0, 0))
            fix = Quaternion(axis.normalized(), tilt - max_tilt)
            pb[hand].matrix = Matrix.Translation(m.translation) @ (fix @ m.to_quaternion()).to_matrix().to_4x4()
            bpy.context.view_layer.update()
            pose[hand] = (pose[hand][0], pb[hand].rotation_quaternion.copy())
    for p in pb:
        p.matrix_basis.identity()
    return math.degrees(worst)


def recoil(rig, poses, cfg, at_frame):
    """Procedural kick (see the module doc). Returns the peak muzzle climb in degrees."""
    hand = cfg['hand']
    foot, head = Vector(cfg['axis'][0]), Vector(cfg['axis'][1])
    settle = float(cfg.get('settle', 0.25))
    pb = rig.pose.bones
    rig.animation_data.action = None
    local_axis = rig.data.bones[hand].matrix_local.to_quaternion().inverted() @ (head - foot).normalized()
    order = [b.name for b in rig.data.bones]  # parents before children
    bones = [b for b in order if b in cfg['bones']]
    up = Vector((0, 0, 1))
    peak = 0.0
    for i, pose in enumerate(poses):
        t = (i - at_frame) / FPS
        if t < -1.0 / FPS or t > settle:
            continue
        env = max(0.0, 1.0 + t * FPS) if t < 0 else (1 - t / settle) ** 2
        for name, (loc, q) in pose.items():
            pb[name].rotation_mode = 'QUATERNION'
            pb[name].rotation_quaternion = q
            pb[name].location = loc if pb[name].parent is None else Vector()
        bpy.context.view_layer.update()
        d = pb[hand].matrix.to_quaternion() @ local_axis
        axis = d.cross(up)
        if axis.length < 1e-6:
            continue
        axis.normalize()
        before = d.angle(up)
        for b in bones:
            m = pb[b].matrix.copy()
            fix = Quaternion(axis, math.radians(float(cfg['bones'][b])) * env)  # + about (dir x up) lifts the muzzle
            pb[b].matrix = Matrix.Translation(m.translation) @ (fix @ m.to_quaternion()).to_matrix().to_4x4()
            bpy.context.view_layer.update()
            pose[b] = (pose[b][0], pb[b].rotation_quaternion.copy())
        climb = math.degrees(before - (pb[hand].matrix.to_quaternion() @ local_axis).angle(up))
        peak = climb if abs(climb) > abs(peak) else peak
    for p in pb:
        p.matrix_basis.identity()
    return peak


def swing_twist(q, axis):
    """Split q into (twist about axis) and swing: q = twist @ swing."""
    p = Vector((q.x, q.y, q.z)).project(axis)
    t = Quaternion((q.w, p.x, p.y, p.z))
    if t.magnitude < 1e-9:
        t = Quaternion()
    t.normalize()
    return t, t.inverted() @ q


def secondary_motion(rig, poses, cfg, loop):
    """Tail-style sway on bones the source clip does not drive (see the module doc)."""
    n = len(poses)
    if n < 2:
        return 0.0
    T = (n - 1) / FPS
    period = float(cfg.get('period', 1.6))
    cycles = max(1, round(T / period)) if loop else T / period
    sway = cfg.get('swayDeg', 8)
    grow = float(cfg.get('grow', 0.35))
    lag = float(cfg.get('lag', 0.7))
    decay = bool(cfg.get('decay', False))
    level = float(cfg.get('level', 0.0))
    RW = rig.matrix_world.to_3x3().normalized()
    bones = cfg['bones']
    if level > 0:
        # the first bone cancels `level` of its parent's pitch/roll (yaw is kept),
        # so a rigid tail trails behind instead of swinging up with a forward lunge
        first = rig.data.bones[bones[0]]
        par = first.parent
        Tr = first.matrix_local.to_quaternion()
        Pr = par.matrix_local.to_quaternion()
        up = Vector((0, 0, 1))
        pb = rig.pose.bones
        rig.animation_data.action = None
        for pose in poses:
            for name, (loc, q) in pose.items():
                pb[name].rotation_mode = 'QUATERNION'
                pb[name].rotation_quaternion = q
                pb[name].location = loc if pb[name].parent is None else Vector()
            bpy.context.view_layer.update()
            D = pb[par.name].matrix.to_quaternion() @ Pr.inverted()
            tw, sw = swing_twist(D, up)
            Dt = tw @ sw.slerp(Quaternion(), level)
            B = Tr.inverted() @ D.inverted() @ Dt @ Tr
            loc, q = pose[bones[0]]
            pose[bones[0]] = (loc, (B @ q).normalized())
        for p in pb:
            p.matrix_basis.identity()
    total = 0.0
    for bi, bone in enumerate(bones):
        b = rig.data.bones[bone]
        up_l = ((RW @ b.matrix_local.to_3x3()).inverted() @ Vector((0, 0, 1))).normalized()
        amp = math.radians(sway[bi] if isinstance(sway, list) else sway * (1 + grow * bi))
        total += math.degrees(amp)
        for i, pose in enumerate(poses):
            t = i / (n - 1)
            ang = amp * math.sin(2 * math.pi * cycles * t - bi * lag)
            if decay:
                ang *= (1 - t) ** 1.5
            loc, q = pose[bone]
            pose[bone] = (loc, (Quaternion(up_l, ang) @ q).normalized())
    return total


def write_action(rig, name, poses, markers, loop):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    rig.animation_data.action = act
    prev = {}
    for i, pose in enumerate(poses):
        for bone, (loc, q) in pose.items():
            pb = rig.pose.bones[bone]
            pb.rotation_mode = 'QUATERNION'
            if bone in prev:
                q = q.copy()
                q.make_compatible(prev[bone])
            prev[bone] = q
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=i)
            if pb.parent is None:
                pb.location = loc
                pb.keyframe_insert('location', frame=i)
    for mname, frame in markers.items():
        act.pose_markers.new(mname).frame = int(round(frame))
    act['loop'] = bool(loop)
    rig.animation_data.action = None
    return act


def ignored_vertices(meshes, bones):
    """{mesh name: vertex indices whose dominant bone is in `bones`}."""
    out = {}
    if not bones:
        return out
    for m in meshes:
        names = {vg.index: vg.name for vg in m.vertex_groups}
        out[m.name] = {v.index for v in m.data.vertices
                       if v.groups and names.get(max(v.groups, key=lambda g: g.weight).group) in bones}
    return out


def lowest_point(meshes, ignore=None):
    dg = bpy.context.evaluated_depsgraph_get()
    low = float('inf')
    for m in meshes:
        ev = m.evaluated_get(dg)
        em = ev.to_mesh()
        mw = m.matrix_world
        skip = (ignore or {}).get(m.name, ())
        for v in em.vertices:
            if v.index in skip:
                continue
            z = (mw @ v.co).z
            if z < low:
                low = z
        ev.to_mesh_clear()
    return low


def ground_clamp(rig, meshes, act, floor, smooth=2, ignore=None):
    """Lift the root on frames where the mesh dips below the floor.

    Mixamo's lying and sitting poses are captured on human proportions; a
    goblin's big head and belly then sink through the ground (sleep, death).
    Measure the real skinned mesh each frame, lift only (never push down),
    and smooth the lift a little so it can't add jitter."""
    rig.animation_data.action = act
    if act.slots:
        rig.animation_data.action_slot = act.slots[0]
    root = next(b for b in rig.data.bones if b.parent is None)
    pb = rig.pose.bones[root.name]
    start, end = (int(round(v)) for v in act.frame_range)
    frames = list(range(start, end + 1))
    lifts = []
    for f in frames:
        bpy.context.scene.frame_set(f)
        lifts.append(max(0.0, floor - lowest_point(meshes, ignore)))
    if smooth > 0:
        lifts = [max(lifts[max(0, i - smooth):i + smooth + 1]) for i in range(len(lifts))]
    to_local = (rig.matrix_world @ root.matrix_local).inverted().to_3x3()
    worst = max(lifts) if lifts else 0.0
    if worst > 1e-5:
        for f, lift in zip(frames, lifts):
            bpy.context.scene.frame_set(f)
            pb.location = pb.location + to_local @ Vector((0, 0, lift))
            pb.keyframe_insert('location', frame=f)
    rig.animation_data.action = None
    return worst


def main():
    args = parse()
    spec = json.loads(Path(args.spec).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS
    rig, rig_objs, stray = import_glb(spec['rig'])
    for a in stray:
        bpy.data.actions.remove(a)
    rig_meshes = [o for o in rig_objs if o.type == 'MESH']
    rig.animation_data_create()
    rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    # clampIgnoreBones: vertices dominated by these bones (a rigid weapon on the
    # hand) never lift the body: a corpse's axe pointing down sinks into the
    # ground instead of propping the whole body up by its length
    clamp_ignore = ignored_vertices(rig_meshes, set(spec.get('clampIgnoreBones', [])))
    floor = lowest_point(rig_meshes, clamp_ignore)  # rest pose: soles of the feet
    sources = spec['candidates'] if isinstance(spec['candidates'], list) else [spec['candidates']]
    imported = []  # [(src_rig, objs, {name: action})], searched in spec order
    for path in sources:
        src_rig, src_objs, acts = import_glb(path)
        src_rig.animation_data_create()
        imported.append((src_rig, src_objs, {a.name: a for a in acts}))
    candidates = [a for _, _, by in imported for a in by.values()]
    src_objs = [o for _, objs, _ in imported for o in objs]
    table = {}
    for name, c in spec['clips'].items():
        src_rig, action = next((r, by[c['source']]) for r, _, by in imported if c['source'] in by)
        lo, hi = c.get('range', [int(action.frame_range[0]), int(action.frame_range[1])])
        speed = float(c.get('speed', 1.0))
        count = max(2, int(round((hi - lo) / speed)) + 1)
        frames = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
        poses = sample(src_rig, action, frames)
        keep = {**spec.get('boneKeep', {}), **c.get('boneKeep', {})}
        for pose in poses:
            for bone, k in keep.items():
                if bone in pose:
                    loc, q = pose[bone]
                    q = q.copy()
                    if q.w < 0:
                        q.negate()  # shortest path from identity
                    pose[bone] = (loc, Quaternion().slerp(q, float(k)))
        if 'yawKeep' in c:
            world_yaw_limit(src_rig, poses, float(c['yawKeep']), c.get('yawRef', 'first'))
        prop = c.get('twoHandProp', spec.get('twoHandProp'))
        gap = two_hand_prop(src_rig, poses, prop) if prop else None
        kick = c.get('recoil')
        climb = None
        if kick:
            climb = recoil(src_rig, poses, kick, (kick['at'] - lo) / speed)
            if prop:
                gap = two_hand_prop(src_rig, poses, {**prop, 'aim': 0.0})
        upright = c.get('uprightProp', spec.get('uprightProp'))
        tilt = upright_prop(src_rig, poses, upright) if upright else None
        loop = bool(c.get('loop', False))
        sec = spec.get('secondary')
        if sec and 'secondary' in c:
            sec = None if c['secondary'] is None else {**sec, **c['secondary']}
        sway = round(secondary_motion(src_rig, poses, sec, loop), 1) if sec else None
        seam = None
        if loop:
            seam = round(close_loop(poses), 2)
        markers = {k: (v - lo) / speed for k, v in c.get('markers', {}).items()}
        act = write_action(rig, name, poses, markers, loop)
        lift = round(ground_clamp(rig, rig_meshes, act, floor, ignore=clamp_ignore), 4) if c.get('groundClamp') else None
        table[name] = {'source': c['source'], 'range': [lo, hi], 'speed': speed, 'loop': loop, 'sourceSeamDeg': seam,
                       'seconds': round((count - 1) / FPS, 3), 'groundLiftMax': lift, 'boneKeep': keep,
                       **({'twoHandProp': prop, 'worstWristGap': round(gap, 4)} if prop else {}),
                       **({'recoil': kick, 'muzzleClimbDeg': round(climb, 1)} if kick else {}),
                       **({'uprightProp': upright, 'worstTiltBeforeDeg': round(tilt, 1)} if upright else {}),
                       **({'secondaryPeakDeg': sway} if sec else {}),
                       'events': {k: round(v / FPS, 3) for k, v in markers.items()}}
    for o in src_objs:
        bpy.data.objects.remove(o)
    for a in candidates:
        bpy.data.actions.remove(a)
    extra = spec.get('extra')
    if extra:
        _, xobjs, xacts = import_glb(extra['from'])
        xby = {a.name: a for a in xacts}
        for new_name, old in extra['actions'].items():
            a = xby[old]
            a.name = new_name
            a.use_fake_user = True
            table[new_name] = {'source': f"{Path(extra['from']).name}:{old}", 'copied': True,
                               'seconds': round((a.frame_range[1] - a.frame_range[0]) / FPS, 3)}
        for a in xacts:
            if a.name not in extra['actions']:
                bpy.data.actions.remove(a)
        for o in xobjs:
            bpy.data.objects.remove(o)
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
    Path(str(out).replace('.glb', '.clips.json')).write_text(
        json.dumps({'spec': args.spec, 'fps': FPS, 'clips': table}, indent=2) + '\n')
    print('POLISHED', json.dumps(table))


if __name__ == '__main__':
    main()
