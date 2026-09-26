# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Poses and clips for the many-legged body plans (arachnid, crustacean,
scorpion), used by rig_critter.py. Pure mathutils, no bpy.

WORLD-SPACE POSING. The quadruped/bird clips in rig_critter.py author local
rotations bone by bone. A crab or spider has 8-10 legs whose feet must stay
planted, so here a pose is authored as:
  rel[bone]  a rotation RELATIVE TO THE PARENT, about axes given in rest
             (rig) space: `turn('body', X, 10)` tips the body about the rig's
             X axis wherever the root has moved it
  ik[leg]    a world target for that leg's foot tip; the leg is solved
  root_loc   world translation of the whole rig (bob, lunge, sink)
`solve()` runs forward kinematics in parent-first order and turns every bone
into a world delta D (posed orientation = D @ rest orientation); `local()`
converts to Blender pose quaternions: L = R^-1 D_parent^-1 D R.

LEG IK. Each leg is 4 bones on joints j0 (root, in the body) .. j4 (tip):
  1. the coxa yaws about the body's up axis at j0 so the leg faces the target
  2. femur (j1-j2) and the rigid distal link (j2-j4: tibia and tarsus keep
     their rest angle) are solved as a two-bone IK in the leg's own bend
     plane: the rest plane carried by the yaw, so the rest target returns
     exactly the rest pose (no pop between an IK clip and an FK one)
Unreachable targets are clamped (the foot stops short) and counted.

GAIT. Legs leg1 (front) .. legN (back) per side walk an ALTERNATING
TETRAPOD: legs with (i + side) even swing together, the others half a cycle
later (L1 R2 L3 R4 against R1 L2 R3 L4), with a duty factor over one half so
some feet always overlap on the ground. Any N works; N = 3 gives an
alternating tripod. In stance the foot slides back along the travel
heading at exactly the stride per cycle (a planted foot under a moving
body); in swing it lifts on an arc and returns. `heading_deg` 0 walks
toward -Y (forward), 90 toward +X (the crab's sideways scuttle).
"""
import math
from mathutils import Vector, Quaternion

X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
SEGS = ('coxa', 'femur', 'tibia', 'tarsus')


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def bump(t, a, b, ramp=0.25):
    if t <= a or t >= b:
        return 0.0
    u = (t - a) / (b - a)
    return smooth(u / ramp) if u < ramp else (smooth((1 - u) / ramp) if u > 1 - ramp else 1.0)


def ramp(t, a, b):
    return smooth((t - a) / max(b - a, 1e-9))


def between(u, v):
    if u.length < 1e-9 or v.length < 1e-9:
        return Quaternion()
    return u.normalized().rotation_difference(v.normalized())


class Rig:
    """Static rest-pose facts: bones in parent-first order, joints, legs."""

    def __init__(self, bones, J, rest_q, anim, height):
        self.bones = bones
        self.J = J
        self.R = rest_q
        self.A = anim
        self.H = height
        self.parent = {b[0]: b[3] for b in bones}
        self.head = {b[0]: J[b[1]].copy() for b in bones}
        self.tailp = {b[0]: J[b[2]].copy() for b in bones}
        self.names = [b[0] for b in bones]
        self.legs = []                           # (i, side)
        for s in ('L', 'R'):
            i = 1
            while f'leg{i}_coxa.{s}' in self.head:
                self.legs.append((i, s))
                i += 1
        self.nleg = max([i for i, _ in self.legs] or [0])
        self.leg_of = {f'leg{i}_coxa.{s}': (i, s) for i, s in self.legs}
        reach = [self.out_h(i, s).length for i, s in self.legs]
        self.reach = sum(reach) / len(reach) if reach else 0.3 * height
        # How far the body can rise with every foot still planted: the legs
        # straighten (femur + distal link) before anything lifts off.
        rises = []
        for i, s in self.legs:
            j1, j2, j4 = self.lj(i, s, 1), self.lj(i, s, 2), self.lj(i, s, 4)
            L = (j2 - j1).length + (j4 - j2).length
            v = j4 - j1
            hz = Vector((v.x, v.y, 0)).length
            rises.append(v.z + math.sqrt(max(L * L - hz * hz, 0.0)))
        self.max_rise = max(0.0, min(rises)) if rises else 0.0
        self.tail = []
        k = 1
        while f'tail_{k}' in self.head:
            self.tail.append(f'tail_{k}')
            k += 1

    def has(self, bn):
        return bn in self.head

    def lj(self, i, s, k):
        return self.J[f'leg{i}_{k}.{s}']

    def out_h(self, i, s):
        """Horizontal vector from the leg's root to its tip (rest)."""
        v = self.lj(i, s, 4) - self.lj(i, s, 0)
        return Vector((v.x, v.y, 0.0))

    def fold_axis(self, i, s):
        """Turning a leg segment by +deg about this axis RAISES its far end."""
        o = self.out_h(i, s)
        return o.cross(Z).normalized() if o.length > 1e-6 else X

    def phase(self, i, s):
        return 0.5 * ((i - 1 + (1 if s == 'R' else 0)) % 2)


class WPose:
    def __init__(self, rig):
        self.rig = rig
        self.rel = {}
        self.ik = {}
        self.root_loc = Vector((0, 0, 0))
        self.scale = {}
        self.clamped = 0
        self.clamp_legs = {}

    # ── authoring helpers ───────────────────────────────────────────────
    def turn(self, bn, axis, deg):
        if deg and self.rig.has(bn):
            self.rel[bn] = Quaternion(axis, math.radians(deg)) @ self.rel.get(bn, Quaternion())

    def aim(self, bn, anchor, dir_rest, amount=1.0):
        """Point bone `bn` along `dir_rest`, a direction in the REST frame of
        bone `anchor` (so it rides the anchor's motion), blended by `amount`
        over whatever the bone would otherwise do. For poses a rotation about
        a fixed axis cannot reach cleanly, e.g. wings folded along the flanks."""
        if self.rig.has(bn) and amount:
            self.aims = getattr(self, 'aims', {})
            self.aims[bn] = (anchor, Vector(dir_rest).normalized(), max(0.0, min(1.0, amount)))

    def leg_turn(self, i, s, seg, deg):
        """Raise (+) / lower (-) one leg segment about its own fold axis."""
        self.turn(f'leg{i}_{seg}.{s}', self.rig.fold_axis(i, s), deg)

    def leg_swing(self, i, s, deg):
        """Swing the whole leg forward (+, toward -Y) about the vertical."""
        self.turn(f'leg{i}_coxa.{s}', Z, -deg if s == 'L' else deg)

    def foot(self, i, s, offset=Vector((0, 0, 0))):
        """IK the leg so its tip sits at rest tip + offset (world)."""
        self.ik[(i, s)] = self.rig.lj(i, s, 4) + offset

    def foot_local(self, i, s, target_rest):
        """IK the leg to a point given in the REST frame of the bone the leg
        hangs from, so it moves with the body (a curl under a body that is
        flipping over)."""
        self.ik_local = getattr(self, 'ik_local', {})
        self.ik_local[(i, s)] = target_rest

    def plant_all(self, except_=()):
        for i, s in self.rig.legs:
            if (i, s) not in self.ik and (i, s) not in except_:
                self.foot(i, s)

    # ── solve ───────────────────────────────────────────────────────────
    def solve(self):
        rig = self.rig
        D, head = {}, {}
        done = set()
        for bn in rig.names:
            p = rig.parent[bn]
            Dp = D[p] if p else Quaternion()
            if p is None:
                head[bn] = rig.head[bn] + self.root_loc
            else:
                head[bn] = head[p] + Dp @ (rig.head[bn] - rig.head[p])
            if bn in done:
                continue
            leg = rig.leg_of.get(bn)
            loc = getattr(self, 'ik_local', {})
            if leg and leg in loc and leg not in self.ik:
                self.ik[leg] = head[p] + Dp @ (loc[leg] - rig.head[p])
            if leg and leg in self.ik:
                self._leg_ik(leg, Dp, head[bn], D)
                done.update(f'leg{leg[0]}_{sg}.{leg[1]}' for sg in SEGS)
                continue
            D[bn] = Dp @ self.rel.get(bn, Quaternion())
            aim = getattr(self, 'aims', {}).get(bn)
            if aim:
                anchor, dr, k = aim
                rest_dir = (rig.tailp[bn] - rig.head[bn]).normalized()
                full = between(D[bn] @ rest_dir, D[anchor] @ dr) @ D[bn]
                D[bn] = D[bn].slerp(full, k)
        self.D, self.heads = D, head
        return D, head

    def _leg_ik(self, leg, Db, j0, D):
        rig = self.rig
        i, s = leg
        T = self.ik[leg]
        j0r, j1r, j2r, j4r = (rig.lj(i, s, k) for k in (0, 1, 2, 4))
        up = Db @ Z
        out = Db @ (j4r - j0r)
        out -= up * out.dot(up)
        tgt = T - j0
        tgt -= up * tgt.dot(up)
        yaw = 0.0
        if out.length > 1e-6 and tgt.length > 1e-6:
            yaw = out.angle(tgt)
            if out.cross(tgt).dot(up) < 0:
                yaw = -yaw
        cx = f'leg{i}_coxa.{s}'
        Dc = Quaternion(up, yaw) @ Db @ self.rel.get(cx, Quaternion())
        j1 = j0 + Dc @ (j1r - j0r)
        flat = rig.A.get('flat_feet', False)
        if flat:
            # A vertebrate foot stays flat on the ground and does NOT turn
            # while it is planted (the leg and body turn over it): the tarsus
            # keeps its rest attitude in the world, and the femur + tibia
            # solve to the ankle above the foot.
            Dfoot = Quaternion()
            j3r = rig.lj(i, s, 3)
            T = T - Dfoot @ (j4r - j3r)
            j4r_eff = j3r
        else:
            j4r_eff = j4r
        a = (j2r - j1r).length
        b = (j4r_eff - j2r).length
        dv = T - j1
        dist = dv.length
        lo, hi = abs(a - b) + 1e-4, a + b - 1e-4
        if dist > hi or dist < lo:
            self.clamped += 1
            over = (dist - hi) if dist > hi else (lo - dist)
            self.clamp_legs[f'leg{i}.{s}'] = max(self.clamp_legs.get(f'leg{i}.{s}', 0.0), over)
            dist = max(lo, min(hi, dist))
        e1 = dv.normalized() if dv.length > 1e-9 else (Dc @ (j4r - j1r)).normalized()
        ur = (j4r_eff - j1r).normalized()
        koff = (j2r - j1r) - ur * (j2r - j1r).dot(ur)
        n = Dc @ koff
        n -= e1 * n.dot(e1)
        if n.length < 1e-6:
            n = up - e1 * up.dot(e1)
        n.normalize()
        x = (a * a - b * b + dist * dist) / (2 * dist)
        y = math.sqrt(max(a * a - x * x, 0.0))
        knee = j1 + e1 * x + n * y
        tip = j1 + e1 * dist
        Df = between(Dc @ (j2r - j1r), knee - j1) @ Dc
        Dd = between(Df @ (j4r_eff - j2r), tip - knee) @ Df
        D[cx] = Dc
        D[f'leg{i}_femur.{s}'] = Df
        D[f'leg{i}_tibia.{s}'] = Dd
        D[f'leg{i}_tarsus.{s}'] = Dfoot if flat else Dd

    def local(self):
        """Blender pose quaternions + the root's local location."""
        D, _ = self.solve()
        rig = self.rig
        q = {}
        for bn in rig.names:
            p = rig.parent[bn]
            Dp = D[p] if p else Quaternion()
            R = rig.R[bn]
            q[bn] = R.inverted() @ Dp.inverted() @ D[bn] @ R
        return q


# ── clip helpers shared by the plans ────────────────────────────────────
def heading(rig):
    a = math.radians(rig.A.get('heading_deg', 0.0))
    return Vector((math.sin(a), -math.cos(a), 0.0))


def gait(p, t, stride=None, lift=None, duty=None, legs=None):
    """Alternating-tetrapod foot targets (see module doc)."""
    rig = p.rig
    A = rig.A
    S = stride if stride is not None else A.get('stride', 0.45) * rig.reach
    Lh = lift if lift is not None else A.get('step_height', 0.3) * rig.reach
    Df = duty if duty is not None else A.get('duty', 0.56)
    hd = heading(rig)
    for i, s in (legs or rig.legs):
        c = (t + rig.phase(i, s)) % 1.0
        if c < Df:
            u = c / Df
            off = hd * (S / 2 - S * u)
        else:
            u = (c - Df) / (1 - Df)
            off = hd * (-S / 2 + S * smooth(u)) + Z * (Lh * math.sin(math.pi * u))
        p.foot(i, s, off)
    return S


def shuffle(p, t, picks, lift_frac=0.18):
    """Idle: a few legs lift and re-plant in place, one at a time."""
    rig = p.rig
    for (i, s), a, b in picks:
        if (i, s) in rig.legs:
            p.foot(i, s, Z * (lift_frac * rig.reach * bump(t, a, b, 0.5)))


def pinch(p, s, deg):
    """Open (+) / close (-) the pincer on side s by turning the dactyl about
    the hinge axis normal to the plane of the two fingers."""
    rig = p.rig
    if not rig.has(f'claw_finger.{s}'):
        return
    g = rig.J[f'dactyl_0.{s}']
    d = rig.J[f'dactyl_1.{s}'] - g
    f = rig.J[f'claw_4.{s}'] - g
    n = d.cross(f)
    if n.length < 1e-9:
        return
    p.turn(f'claw_finger.{s}', n.normalized(), -deg)


def claw_axes(rig, s):
    """(outward-horizontal fold axis, lateral axis) for a claw."""
    o = rig.J[f'claw_4.{s}'] - rig.J[f'claw_0.{s}']
    o = Vector((o.x, o.y, 0.0))
    fold = o.cross(Z).normalized() if o.length > 1e-6 else X
    return fold


def buckle_feet(p, drop, draw, splay=0.25, draw_in=0.45):
    """Dying legs: the feet stay ON the ground (world targets) and slide
    outward as the body drops onto them, then draw in a little. Folding the
    legs by FK instead drives the tibiae through the ground, and the ground
    solve then floats the whole corpse above it."""
    for i, s in p.rig.legs:
        j0, j4 = p.rig.lj(i, s, 0), p.rig.lj(i, s, 4)
        out = Vector((j4.x - j0.x, j4.y - j0.y, 0.0))
        p.foot(i, s, out * (splay * drop) - out * (draw_in * draw))


def curl_legs(p, amt, femur=35, tibia=-95, tarsus=-45):
    for i, s in p.rig.legs:
        p.leg_turn(i, s, 'femur', femur * amt)
        p.leg_turn(i, s, 'tibia', tibia * amt)
        p.leg_turn(i, s, 'tarsus', tarsus * amt)


def sgn(s):
    return 1.0 if s == 'L' else -1.0


# ── crustacean ──────────────────────────────────────────────────────────
def crab_idle(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    p.root_loc = Vector((0, 0, 0.012 * rig.H * (0.5 + 0.5 * math.sin(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink      # two breaths
    p.turn('body', Y, 1.2 * math.sin(w))
    n = rig.nleg
    shuffle(p, t, [((1, 'L'), 0.08, 0.22), ((max(n - 1, 1), 'R'), 0.30, 0.44),
                   ((min(3, n), 'L'), 0.56, 0.70), ((2, 'R'), 0.76, 0.90)])
    p.plant_all()
    for s in ('L', 'R'):
        ph = 0.0 if s == 'L' else 0.9
        p.turn(f'claw_arm.{s}', X, 4 * math.sin(w + ph))
        p.turn(f'claw_fore.{s}', claw_axes(rig, s), 3 * math.sin(w + ph + 0.8))
        # one lazy open-and-shut per side, at different times
        a = 0.18 if s == 'L' else 0.58
        pinch(p, s, 22 * bump(t, a, a + 0.26, 0.4))
    for s in ('L', 'R'):
        tw = bump(t, 0.40 if s == 'L' else 0.43, 0.52 if s == 'L' else 0.55, 0.3)
        p.turn(f'eye.{s}', Y, sgn(s) * 10 * tw + 3 * math.sin(w))
        p.turn(f'eye.{s}', X, -6 * bump(t, 0.80, 0.95, 0.3))


def crab_walk(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    S = gait(p, t)
    p.root_loc = Vector((0, 0, rig.A.get('bob', 0.012) * rig.H * (0.5 + 0.5 * math.cos(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink
    hd = heading(rig)
    # a sideways scuttle rolls the carapace, a forward walk pitches it
    p.turn('body', Y, 2.0 * math.sin(w) * abs(hd.x))
    p.turn('body', X, 1.5 * math.sin(w) * abs(hd.y))
    p.turn('body', Z, 2.0 * math.sin(w) * abs(hd.y))
    for s in ('L', 'R'):
        ph = 0.0 if s == 'L' else math.pi
        p.turn(f'claw_arm.{s}', X, 5 * math.sin(w + ph))
        pinch(p, s, 6 + 6 * math.sin(2 * w + ph))
        p.turn(f'eye.{s}', X, 3 * math.sin(2 * w))
    return {'stride_world': S}


def crab_attack(t, p):
    """Claw snap: wind up (claws back, pincers wide), strike forward-down,
    SNAP shut on the hit, hold, recover. The feet stay planted."""
    rig = p.rig
    wind = ramp(t, 0.0, 0.32) * (1 - ramp(t, 0.36, 0.46))
    strike = ramp(t, 0.34, 0.46) * (1 - ramp(t, 0.62, 0.95))
    open_ = ramp(t, 0.05, 0.30) * (1 - ramp(t, 0.44, 0.49))
    shut = ramp(t, 0.44, 0.49) * (1 - ramp(t, 0.70, 0.95))
    p.root_loc = Vector((0, -0.05 * rig.H * strike + 0.02 * rig.H * wind, -0.02 * rig.H * strike))
    p.turn('body', X, 6 * strike - 4 * wind)
    for s in ('L', 'R'):
        lag = 0.0 if s == 'R' else 0.03
        st = ramp(t, 0.34 + lag, 0.46 + lag) * (1 - ramp(t, 0.62, 0.95))
        p.turn(f'claw_arm.{s}', X, -18 * wind + 34 * st)
        p.turn(f'claw_fore.{s}', X, -10 * wind + 26 * st)
        p.turn(f'claw_palm.{s}', X, 10 * st)
        pinch(p, s, 38 * open_ - 6 * shut)
        p.turn(f'eye.{s}', X, -8 * strike)
    p.plant_all()


def crab_death(t, p):
    rig = p.rig
    flinch = bump(t, 0.0, 0.2, 0.4)
    fall = ramp(t, 0.12, 0.62)
    p.root_loc = Vector((0, 0, 0.03 * rig.H * flinch - rig.A.get('death_sink', 0.25) * rig.H * fall))
    p.turn('body', Y, 8 * fall)
    p.turn('body', X, 4 * fall)
    buckle_feet(p, fall, 0.3 * ramp(t, 0.6, 0.9) + 0.04 * math.sin(math.pi * 3 * ramp(t, 0.6, 0.9)) * bump(t, 0.6, 0.95))
    for s in ('L', 'R'):
        p.turn(f'claw_arm.{s}', claw_axes(rig, s), -rig.A.get('death_claw_drop', 25) * fall)
        p.turn(f'claw_fore.{s}', claw_axes(rig, s), -20 * fall)
        pinch(p, s, 25 * fall)
        p.turn(f'eye.{s}', X, 60 * fall)


def crab_threat(t, p):
    """special_1: rise on the legs, claws high and wide, three clacks."""
    rig = p.rig
    up = ramp(t, 0.0, 0.25) * (1 - ramp(t, 0.8, 1.0))
    p.root_loc = Vector((0, 0, min(rig.A.get('rise', 0.06) * rig.H, 0.8 * rig.max_rise) * up))
    p.turn('body', X, -5 * up)
    for s in ('L', 'R'):
        p.turn(f'claw_arm.{s}', claw_axes(rig, s), 20 * up)
        p.turn(f'claw_arm.{s}', X, -8 * up)
        clack = 0.5 + 0.5 * math.cos(2 * math.pi * 3 * ramp(t, 0.25, 0.8))
        pinch(p, s, up * (8 + 30 * clack))
    p.plant_all()


# ── arachnid ────────────────────────────────────────────────────────────
def fangs(p, deg):
    for s in ('L', 'R'):
        p.turn(f'fang.{s}', Y, -sgn(s) * deg)
        p.turn(f'fang.{s}', X, -0.5 * deg)


def palps(p, deg, ph=0.0):
    for s in ('L', 'R'):
        p.turn(f'palp.{s}', X, deg * (1 if s == 'L' else -1) * math.sin(ph) if ph else deg)


def spider_idle(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    breath = 0.5 - 0.5 * math.cos(2 * w)
    p.root_loc = Vector((0, 0, 0.008 * rig.H * (0.5 + 0.5 * math.sin(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink
    p.turn('abdomen', X, 2.5 * math.sin(2 * w))
    p.scale['abdomen'] = 1.0 + 0.03 * breath
    n = rig.nleg
    shuffle(p, t, [((1, 'L'), 0.06, 0.20), ((n, 'R'), 0.26, 0.40), ((2, 'R'), 0.50, 0.64),
                   ((1, 'R'), 0.70, 0.84)], lift_frac=0.15)
    p.plant_all()
    p.turn('body', Z, 3 * math.sin(w))
    for s in ('L', 'R'):
        p.turn(f'palp.{s}', X, 8 * math.sin(3 * w + (0 if s == 'L' else 1.4)))
    fangs(p, 6 * bump(t, 0.44, 0.56, 0.4))


def spider_walk(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    S = gait(p, t)
    p.root_loc = Vector((0, 0, rig.A.get('bob', 0.01) * rig.H * (0.5 + 0.5 * math.cos(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink
    p.turn('body', Z, 2.5 * math.sin(w))
    p.turn('abdomen', X, 3 * math.cos(2 * w - 0.6))
    p.turn('abdomen', Z, -3 * math.sin(w - 0.6))
    for s in ('L', 'R'):
        p.turn(f'palp.{s}', X, 10 * math.sin(w + (0 if s == 'L' else math.pi)))
    return {'stride_world': S}


def spider_attack(t, p):
    """Rear up on the back legs with the front pair raised and fangs spread,
    lunge forward-down and bite (hit), recover."""
    rig = p.rig
    rear = ramp(t, 0.0, 0.34) * (1 - ramp(t, 0.36, 0.48))
    lunge = ramp(t, 0.36, 0.48) * (1 - ramp(t, 0.62, 0.96))
    p.root_loc = Vector((0, -0.10 * rig.H * lunge + 0.03 * rig.H * rear,
                         min(0.05 * rig.H, 0.8 * rig.max_rise) * rear - 0.03 * rig.H * lunge))
    p.turn('body', X, -rig.A.get('rear_deg', 22) * rear + 10 * lunge)
    p.turn('abdomen', X, 14 * rear - 6 * lunge)
    front = [(i, s) for i, s in rig.legs if i <= 2]
    for i, s in front:
        k = 1.0 if i == 1 else 0.55
        p.leg_turn(i, s, 'femur', k * (50 * rear + 10 * lunge))
        p.leg_turn(i, s, 'tibia', -k * 25 * rear)
        p.leg_swing(i, s, 10 * rear)
    p.plant_all(except_=front)
    fangs(p, 22 * ramp(t, 0.1, 0.34) * (1 - ramp(t, 0.44, 0.5)) - 6 * bump(t, 0.46, 0.7))
    for s in ('L', 'R'):
        p.turn(f'palp.{s}', X, -25 * rear)


def spider_collapse(t, p):
    """Heavy death (the queen): the legs buckle outward, the body drops onto
    the ground and rolls a little to one side, the abdomen slumps, then the
    legs draw in with a last twitch. Nothing lifts into the air."""
    rig = p.rig
    drop = ramp(t, 0.08, 0.45)
    roll = rig.A.get('collapse_roll', 18) * ramp(t, 0.2, 0.6)
    p.root_loc = Vector((0, 0, -rig.A.get('collapse_drop', 0.35) * rig.H * drop))
    p.turn('root', Y, roll)
    p.turn('body', X, 6 * drop)
    p.turn('abdomen', X, -8 * ramp(t, 0.2, 0.7))
    curl = 0.35 * ramp(t, 0.55, 0.9) + 0.04 * math.sin(math.pi * 4 * ramp(t, 0.6, 0.95)) * bump(t, 0.6, 0.98)
    for i, s in rig.legs:
        j0, j4 = rig.lj(i, s, 0), rig.lj(i, s, 4)
        out = Vector((j4.x - j0.x, j4.y - j0.y, 0.0))
        # WORLD targets on the ground: the feet stay down while the body
        # drops and rolls over them (body-frame targets would ride the roll
        # and lift one side's feet into the air)
        splay = out * (0.25 * drop)                            # buckle outward
        draw = -out * (0.45 * curl)                            # then draw in
        p.foot(i, s, splay + draw)
    fangs(p, 12 * bump(t, 0.0, 0.3))


def spider_death(t, p):
    rig = p.rig
    if rig.A.get('death') == 'collapse':
        return spider_collapse(t, p)
    flip = rig.A.get('death', 'flip') == 'flip'
    curl = ramp(t, 0.05, 0.55) + 0.06 * math.sin(math.pi * 4 * ramp(t, 0.55, 0.9)) * bump(t, 0.55, 0.95)
    # each foot is drawn in under the body (in the body's own frame, so the
    # curl rides the flip): knees up, tips tucked, the classic dead spider
    tuck = rig.A.get('death_tuck', 0.45)
    for i, s in rig.legs:
        j0, j4 = rig.lj(i, s, 0), rig.lj(i, s, 4)
        under = Vector((j0.x * (1 - tuck), j0.y + 0.3 * (j4.y - j0.y), j0.z - 0.25 * rig.reach))
        p.foot_local(i, s, j4.lerp(under, min(1.0, curl)))
    if flip:
        a = 180 * ramp(t, 0.15, 0.6)
        p.turn('root', Y, a)
    else:
        p.root_loc = Vector((0, 0, -0.3 * rig.H * ramp(t, 0.1, 0.5)))
    p.turn('abdomen', X, -10 * curl)
    fangs(p, 12 * bump(t, 0.0, 0.3))


def spider_web(t, p):
    """special_1 (web cast / web spit): crouch, then heave the abdomen up so
    the spinneret aims up and back, and snap it (release) while the front
    rears and the fangs spread (a spitting spider spits from its fangs;
    either reads at game size as a web shot)."""
    rig = p.rig
    crouch = bump(t, 0.0, 0.3, 0.5)
    heave = ramp(t, 0.2, 0.45) * (1 - ramp(t, 0.7, 1.0))
    snap = bump(t, 0.44, 0.60, 0.3)
    p.root_loc = Vector((0, 0, -0.03 * rig.H * crouch + min(0.03 * rig.H, 0.8 * rig.max_rise) * heave))
    p.turn('body', X, -10 * heave + 4 * crouch)
    p.turn('abdomen', X, rig.A.get('web_heave', 40) * heave + 12 * snap)
    p.plant_all()
    fangs(p, 14 * heave - 4 * snap)


def spider_summon(t, p):
    """special_2 (the queen's summon): rise with the front legs raised and
    spread, three shudders, slam the front legs down (release)."""
    rig = p.rig
    up = ramp(t, 0.0, 0.3) * (1 - ramp(t, 0.62, 0.72))
    slam = bump(t, 0.66, 0.9, 0.3)
    shud = math.sin(2 * math.pi * 6 * t) * bump(t, 0.28, 0.62)
    p.root_loc = Vector((0, 0.02 * rig.H * up, min(0.06 * rig.H, 0.8 * rig.max_rise) * up - 0.03 * rig.H * slam))
    p.turn('body', X, -16 * up + 6 * slam)
    p.turn('abdomen', X, 10 * up + 4 * shud)
    front = [(i, s) for i, s in rig.legs if i <= 2]
    for i, s in front:
        k = 1.0 if i == 1 else 0.6
        p.leg_turn(i, s, 'femur', k * (60 * up + 3 * shud))
        p.leg_swing(i, s, -18 * up)
    p.plant_all(except_=front)
    fangs(p, 18 * up)


# ── scorpion ────────────────────────────────────────────────────────────
def tail_curve(p, base, curl=0.0, sway=0.0, ph=0.0):
    """Pose a scorpion tail curled over the back. `base` > 0 swings the whole
    tail forward from its root (the first two segments), `curl` > 0 tightens
    the curl of the rest (spread evenly over it), < 0 opens it so the sting
    points forward. A strike is base forward with the curl opened; a wind-up
    is the reverse. `sway` swings it side to side with a travelling phase."""
    tail = p.rig.tail
    n = len(tail)
    for k, bn in enumerate(tail):
        if k < 2:
            p.turn(bn, X, base * (0.6 if k == 0 else 0.4))
        elif n > 2:
            p.turn(bn, X, curl / (n - 2))
        if sway:
            p.turn(bn, Z, sway * (0.5 + 0.15 * k) * math.sin(ph - 0.6 * k))


def scorp_idle(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    p.root_loc = Vector((0, 0, 0.008 * rig.H * (0.5 + 0.5 * math.sin(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink
    n = rig.nleg
    shuffle(p, t, [((1, 'R'), 0.1, 0.24), ((n, 'L'), 0.34, 0.48), ((2, 'L'), 0.62, 0.76)], lift_frac=0.15)
    p.plant_all()
    tail_curve(p, 3 * math.sin(w), 4 * math.sin(w + 1.0), sway=4, ph=w)
    for s in ('L', 'R'):
        p.turn(f'claw_arm.{s}', Z, sgn(s) * 4 * math.sin(w + (0 if s == 'L' else 1)))
        a = 0.2 if s == 'L' else 0.55
        pinch(p, s, 20 * bump(t, a, a + 0.22, 0.4))


def scorp_walk(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    S = gait(p, t)
    p.root_loc = Vector((0, 0, rig.A.get('bob', 0.008) * rig.H * (0.5 + 0.5 * math.cos(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink
    p.turn('body', Z, 2 * math.sin(w))
    tail_curve(p, 2 * math.sin(2 * w), 3 * math.sin(2 * w + 0.8), sway=5, ph=w)
    for s in ('L', 'R'):
        # the claws swing side to side (and lift a touch), never down: they
        # rest on the ground
        p.turn(f'claw_arm.{s}', Z, sgn(s) * 4 * math.sin(w + (0 if s == 'L' else math.pi)))
        p.turn(f'claw_arm.{s}', claw_axes(rig, s), 3 * (0.5 + 0.5 * math.sin(w + (0 if s == 'L' else math.pi))))
    return {'stride_world': S}


def scorp_attack(t, p):
    """Pincer grab (grab), then the tail strikes over the back (hit)."""
    rig = p.rig
    reach = ramp(t, 0.05, 0.25) * (1 - ramp(t, 0.8, 1.0))
    grab = ramp(t, 0.28, 0.34)
    cock = ramp(t, 0.2, 0.42) * (1 - ramp(t, 0.44, 0.52))
    sting = ramp(t, 0.46, 0.56) * (1 - ramp(t, 0.66, 0.96))
    p.root_loc = Vector((0, -0.04 * rig.H * reach, 0.0))
    p.turn('body', X, -4 * cock + 5 * sting)
    for s in ('L', 'R'):
        p.turn(f'claw_arm.{s}', Z, -sgn(s) * 12 * reach)
        p.turn(f'claw_fore.{s}', Z, sgn(s) * 10 * reach)
        pinch(p, s, 30 * reach * (1 - grab) - 4 * grab * (1 - ramp(t, 0.8, 1.0)))
    tail_curve(p, -rig.A.get('cock_deg', 12) * cock + rig.A.get('sting_base_deg', 80) * sting,
               18 * cock - rig.A.get('sting_open_deg', 55) * sting)
    p.plant_all()


def scorp_death(t, p):
    rig = p.rig
    fall = ramp(t, 0.1, 0.6)
    p.root_loc = Vector((0, 0, -rig.A.get('death_sink', 0.3) * rig.H * fall))
    p.turn('body', Y, 10 * fall)
    buckle_feet(p, fall, 0.3 * ramp(t, 0.6, 0.9))
    # the tail curls in tighter and the sting settles onto the back (a
    # dead scorpion's curl); swinging it back and down instead straightened
    # it into the air or drove the sting into the ground
    tail_curve(p, -rig.A.get('death_tail', -12) * ramp(t, 0.2, 0.8), -rig.A.get('death_tail_open', -35) * ramp(t, 0.25, 0.85),
               sway=10 * fall, ph=1.0)
    for s in ('L', 'R'):
        # the claws sag outward, not down through the ground they rest on
        p.turn(f'claw_arm.{s}', Z, sgn(s) * 12 * fall)
        pinch(p, s, 20 * fall)


def scorp_threat(t, p):
    """special_1: claws spread and open, tail raised and quivering (release)."""
    rig = p.rig
    up = ramp(t, 0.0, 0.25) * (1 - ramp(t, 0.8, 1.0))
    q = math.sin(2 * math.pi * 9 * t) * bump(t, 0.25, 0.8)
    p.root_loc = Vector((0, 0, min(0.04 * rig.H, 0.8 * rig.max_rise) * up))
    p.turn('body', X, -6 * up)
    for s in ('L', 'R'):
        p.turn(f'claw_arm.{s}', Z, sgn(s) * 14 * up)
        pinch(p, s, 30 * up)
    tail_curve(p, 10 * up + 2 * q, -12 * up + 3 * q, sway=3 * up, ph=2 * math.pi * 9 * t)
    p.plant_all()


# ── vertebrates: sprawl (salamander) and drake ──────────────────────────
def spine_of(rig):
    return [n for n in rig.names if n.startswith('spine')]


def neck_of(rig):
    return [n for n in rig.names if n.startswith('neck')] + (['head'] if rig.has('head') else [])


def bend_spine(p, deg, node=0.5):
    """Lateral S-bend about Z with a node part-way along the spine: the hips
    yaw one way, the chest the other (a salamander's standing wave)."""
    sp = spine_of(p.rig)
    if not sp:
        return
    p.turn(sp[0], Z, deg)
    for bn in sp[1:]:
        p.turn(bn, Z, -2 * deg / max(len(sp) - 1, 1))


def tail_wave(p, amp, ph, pitch=0.0, k0=0.6):
    for k, bn in enumerate(p.rig.tail):
        p.turn(bn, Z, amp * (k0 + 0.2 * k) * math.sin(ph - 0.7 * (k + 1)))
        if pitch:
            p.turn(bn, X, pitch)


def neck_turn(p, axis, deg, split=(0.6, 0.4)):
    nk = neck_of(p.rig)
    if not nk:
        return
    body, head = nk[:-1], nk[-1]
    for bn in body:
        p.turn(bn, axis, split[0] * deg / max(len(body), 1))
    p.turn(head, axis, split[1] * deg)


def sal_idle(t, p):
    rig = p.rig
    w = 2 * math.pi * t
    p.root_loc = Vector((0, 0, 0.01 * rig.H * (0.5 + 0.5 * math.sin(2 * w))))   # bob UP only: a belly or claws resting on the ground must not sink
    for bn in spine_of(rig):
        p.turn(bn, X, 0.6 * math.sin(2 * w))
    look = 18 * bump(t, 0.08, 0.36) - 14 * bump(t, 0.62, 0.9)
    neck_turn(p, Z, look)
    neck_turn(p, X, -6 * bump(t, 0.4, 0.55, 0.4))              # a taste of the air
    tail_wave(p, 5, w)
    n = rig.nleg
    shuffle(p, t, [((1, 'L'), 0.45, 0.58), ((n, 'R'), 0.7, 0.82)], lift_frac=0.12)
    p.plant_all()


def sal_walk(t, p):
    """Walking trot (diagonal pairs) with the lateral standing wave in the
    spine and a travelling wave down the tail; the head counter-yaws so it
    stays on line."""
    rig = p.rig
    w = 2 * math.pi * t
    S = gait(p, t)
    A = rig.A.get('undulate_deg', 14)
    # Phase: the trunk is concave on the side whose forelimb is retracted and
    # hindlimb protracted (the limbs on the concave side come together); for
    # this gait's phases that puts the left-concave peak at t ~ 0.53.
    u = w + rig.A.get('undulate_phase', 1.38)
    bend_spine(p, A * math.sin(u))
    neck_turn(p, Z, 0.8 * A * math.sin(u))
    tail_wave(p, rig.A.get('tail_sway_deg', 16), u)
    # a walking salamander pushes its belly up off the ground (and the
    # runtime crossfades from idle): without the raise the belly and the leg
    # roots dig in and the ground solve floats the feet off the ground
    raise_ = rig.A.get('walk_raise', 0.05) * rig.H
    p.root_loc = Vector((0, 0, raise_ + 0.008 * rig.H * (0.5 + 0.5 * math.cos(2 * w))))
    return {'stride_world': S}


def sal_attack(t, p):
    rig = p.rig
    wind = ramp(t, 0.0, 0.3) * (1 - ramp(t, 0.34, 0.44))
    snap = ramp(t, 0.36, 0.46) * (1 - ramp(t, 0.6, 0.95))
    p.root_loc = Vector((0, -0.08 * rig.H * snap + 0.03 * rig.H * wind, 0.02 * rig.H * wind))
    neck_turn(p, X, -25 * wind + 22 * snap)
    for bn in spine_of(rig):
        p.turn(bn, X, -3 * wind)
    tail_wave(p, 10 * (wind + snap), 2 * math.pi * t)
    p.plant_all()


def sal_death(t, p):
    rig = p.rig
    flop = ramp(t, 0.1, 0.55)
    # roll half onto its side (reads at game size; the ground solve lays the
    # flank on the ground), with a small settle-back
    roll = rig.A.get('death_roll', 28) * (ramp(t, 0.12, 0.55) - 0.08 * bump(t, 0.55, 0.85, 0.5))
    p.turn('root', Y, roll)
    p.root_loc = Vector((0, 0, -0.2 * rig.H * flop))
    for i, s in rig.legs:
        p.leg_turn(i, s, 'femur', 25 * flop)
        p.leg_turn(i, s, 'tibia', 15 * flop)
    neck_turn(p, X, 20 * ramp(t, 0.3, 0.8))
    neck_turn(p, Z, 25 * ramp(t, 0.3, 0.8))
    bend_spine(p, 10 * ramp(t, 0.2, 0.7))
    for k, bn in enumerate(rig.tail):
        p.turn(bn, Z, 14 * ramp(t, 0.3, 0.9))
    tw = bump(t, 0.55, 0.85, 0.3) * math.sin(2 * math.pi * 5 * t)
    tail_wave(p, 4 * tw, 0.0)


def sal_spit(t, p):
    """special_1 (the magma spit): raise the head, swell, spit (release)."""
    rig = p.rig
    up = ramp(t, 0.0, 0.35) * (1 - ramp(t, 0.75, 1.0))
    spit = bump(t, 0.42, 0.62, 0.3)
    neck_turn(p, X, -30 * up + 25 * spit)
    for bn in spine_of(rig)[-1:]:
        p.turn(bn, X, -6 * up)
    tail_wave(p, 6 * up, 2 * math.pi * 2 * t)
    p.root_loc = Vector((0, 0.02 * rig.H * spit, 0))
    p.plant_all()


def flap(p, s, deg):
    """Raise (+) / lower (-) a wing at the shoulder (about the body's long axis)."""
    p.turn(f'wing_arm.{s}', Y, -sgn(s) * deg)


def wing_pose(p, s, arm, fore, hand, sweep=0.0, fsweep=0.0):
    p.turn(f'wing_arm.{s}', Y, -sgn(s) * arm)
    p.turn(f'wing_fore.{s}', Y, -sgn(s) * fore)
    p.turn(f'wing_hand.{s}', Y, -sgn(s) * hand)
    if sweep:
        p.turn(f'wing_arm.{s}', Z, sgn(s) * sweep)
    if fsweep:
        p.turn(f'wing_fore.{s}', Z, sgn(s) * fsweep)
        p.turn(f'wing_hand.{s}', Z, sgn(s) * 0.6 * fsweep)


def wingbeat(p, ph, amp, bias=0.0, fold=18.0):
    """One stroke per 2*pi of ph: shoulder leads, forearm and hand lag, the
    forearm sweeps back (folds) on the upstroke and spreads on the down."""
    for s in ('L', 'R'):
        up = 0.5 + 0.5 * math.sin(ph + 0.9)        # 1 at the top of the recovery
        wing_pose(p, s, bias + amp * math.cos(ph), 0.45 * amp * math.cos(ph - 0.7),
                  0.35 * amp * math.cos(ph - 1.3), sweep=0.0, fsweep=fold * up)


def tuck_legs(p, amt):
    for i, s in p.rig.legs:
        front = i == 1
        p.turn(f'leg{i}_femur.{s}', X, (35 if front else 50) * amt)
        p.turn(f'leg{i}_tibia.{s}', X, (-70 if front else 25) * amt)
        p.turn(f'leg{i}_tarsus.{s}', X, (40 if front else 30) * amt)


def drake_idle(t, p):
    """Hover: two wingbeats per loop, legs tucked, neck counter-bobbing, the
    tail swaying. Root translation is left to the runtime's flying bob."""
    rig = p.rig
    ph = 2 * math.pi * rig.A.get('beats', 2) * t
    wingbeat(p, ph, rig.A.get('flap_deg', 38), bias=rig.A.get('flap_bias', 0.0))
    tuck_legs(p, 1.0)
    for bn in spine_of(rig):
        p.turn(bn, X, 1.5 * math.sin(ph + 0.5))
    neck_turn(p, X, 6 * math.sin(ph + 2.2))
    tail_wave(p, 6, 2 * math.pi * t, pitch=1.5 * math.sin(ph + 1.5))
    # no root translation: the runtime's flyingBob owns hover movement
    # (tail and secondary motion)


def drake_fly(t, p):
    """Fly-move: body pitched nose-down into the travel, bigger faster
    strokes (three per loop), legs trailing, tail streaming."""
    rig = p.rig
    ph = 2 * math.pi * rig.A.get('move_beats', 3) * t
    lean = rig.A.get('lean_deg', 14)
    for bn in spine_of(rig)[:1]:
        p.turn(bn, X, lean)
    wingbeat(p, ph, rig.A.get('flap_deg', 38) * 1.2, fold=26)
    tuck_legs(p, 1.2)
    neck_turn(p, X, -0.8 * lean + 4 * math.sin(ph + 2.0))
    tail_wave(p, 5, 2 * math.pi * t, pitch=-0.25 * lean)


def drake_breath(t, p):
    """Ranged breath: rear the head back and up (windup), thrust it forward
    and down as the breath goes (release), hold, recover; the wings keep
    beating, spreading wide to brake on the release."""
    rig = p.rig
    wind = ramp(t, 0.05, 0.36) * (1 - ramp(t, 0.38, 0.46))
    rel = ramp(t, 0.38, 0.48) * (1 - ramp(t, 0.72, 0.98))
    ph = 2 * math.pi * 2 * t
    wingbeat(p, ph, rig.A.get('flap_deg', 38) * (1 - 0.4 * rel), bias=10 * rel)
    tuck_legs(p, 1.0)
    neck_turn(p, X, -30 * wind + 34 * rel)
    for bn in spine_of(rig):
        p.turn(bn, X, -4 * wind + 3 * rel)
    tail_wave(p, 6, 2 * math.pi * t, pitch=6 * rel)
    p.root_loc = Vector((0, 0.03 * rig.H * rel, 0.0))     # recoil only


def drake_death(t, p):
    """Wings stop and go limp, the body drops and rolls a little onto its
    side, neck and tail go slack; the ground solve lays it on the ground.
    The wings end spread flat on the ground (the silhouette that reads from
    the RTS camera); folding them by aiming the spars stretched the Meshy
    membrane badly. (In game the renderer's flyer crash drops the unit from
    altitude; this clip is what the body does on the way down.)"""
    rig = p.rig
    crumple = ramp(t, 0.0, 0.35)
    roll = rig.A.get('death_roll', 25) * ramp(t, 0.15, 0.7)
    p.turn('root', Y, roll)
    p.turn('root', X, 10 * ramp(t, 0.1, 0.5))
    for s in ('L', 'R'):
        # a last weak half-beat, then the wing drapes down and crumples a bit
        beat = 18 * math.sin(math.pi * ramp(t, 0.0, 0.3))
        wing_pose(p, s, beat - 14 * crumple, -8 * crumple, -10 * crumple, sweep=10 * crumple, fsweep=18 * crumple)
    tuck_legs(p, 1.0 - 0.6 * ramp(t, 0.4, 0.9))
    neck_turn(p, X, 25 * ramp(t, 0.2, 0.8))
    neck_turn(p, Z, 20 * ramp(t, 0.3, 0.9))
    for bn in rig.tail:
        p.turn(bn, Z, -8 * ramp(t, 0.3, 0.9))


# name, frames, fn, loop, events (clip fraction)
CLIPS = {
    'crustacean': [('idle', 120, crab_idle, True, {}), ('walk', 24, crab_walk, True, {}),
                   ('attack', 30, crab_attack, False, {'windup': 0.3, 'hit': 0.47}),
                   ('death', 40, crab_death, False, {}),
                   ('special_1', 45, crab_threat, False, {'release': 0.5})],
    'arachnid': [('idle', 120, spider_idle, True, {}), ('walk', 20, spider_walk, True, {}),
                 ('attack', 30, spider_attack, False, {'windup': 0.34, 'hit': 0.46}),
                 ('death', 40, spider_death, False, {}),
                 ('special_1', 36, spider_web, False, {'release': 0.5}),
                 ('special_2', 48, spider_summon, False, {'release': 0.72})],
    'scorpion': [('idle', 120, scorp_idle, True, {}), ('walk', 24, scorp_walk, True, {}),
                 ('attack', 36, scorp_attack, False, {'grab': 0.32, 'hit': 0.54}),
                 ('death', 40, scorp_death, False, {}),
                 ('special_1', 45, scorp_threat, False, {'release': 0.5})],
    'sprawl': [('idle', 120, sal_idle, True, {}), ('walk', 30, sal_walk, True, {}),
               ('attack', 30, sal_attack, False, {'windup': 0.3, 'hit': 0.44}),
               ('death', 40, sal_death, False, {}),
               ('special_1', 40, sal_spit, False, {'release': 0.5})],
    'drake': [('idle', 40, drake_idle, True, {}), ('walk', 30, drake_fly, True, {}),
              ('attack', 36, drake_breath, False, {'windup': 0.36, 'release': 0.45}),
              ('death', 40, drake_death, False, {})],
}


def clips_for(plan, anim):
    out = []
    for name, frames, fn, loop, ev in CLIPS[plan]:
        frames = int(anim.get(f'{name}_frames', frames))
        out.append((name, frames, fn, loop, ev))
    keep = anim.get('clips')
    return [c for c in out if not keep or c[0] in keep]


def allowed_bones(label, names):
    """Bones a vertex of this fit_arthropod label may be skinned to: its own
    limb's chain plus the trunk it hangs from. Heat weights on a crab bleed
    from one leg into the next; this confines them."""
    trunk = {n for n in names if n in ('body', 'abdomen', 'spinneret') or n.startswith('spine')}
    first = ('_coxa.', 'claw_arm.', 'eye.', 'fang.', 'palp.', 'wing_arm.')
    if label is None:
        return trunk | {n for n in names if any(f in n for f in first) or n in ('neck1', 'tail_1')}
    if label.startswith('leg'):
        i, s = label[3:].split('.')
        return {n for n in names if n.startswith(f'leg{i}_') and n.endswith('.' + s)} | trunk
    kind, _, s = label.partition('.')
    if kind == 'claw':
        return {f'claw_arm.{s}', f'claw_fore.{s}', f'claw_palm.{s}'} | trunk
    if kind == 'dactyl':
        return {f'claw_finger.{s}'}
    if kind == 'tail':
        return {n for n in names if n.startswith('tail_')} | trunk
    if kind == 'head':
        return {n for n in names if n.startswith('neck') or n == 'head'} | trunk
    if kind == 'wing':
        return {n for n in names if n.startswith('wing_') and n.endswith('.' + s)} | trunk
    return {n for n in names if n.split('.')[0].rstrip('0123456789') == kind and n.endswith('.' + s)} | trunk
