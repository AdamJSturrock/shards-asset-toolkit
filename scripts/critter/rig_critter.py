# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Rig and animate a static Meshy critter in Blender (workflow steps 3-6).

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/rig_critter.py -- \
  --critter cow --out-dir output/critter-authoring/cow/v1

Meshy auto-rigging is humanoid-only, so the animals are rigged here:
  1. import the profile's `source` GLB (or --input) and normalise it to
     rig space exactly as prep_critter.py does (optional profile yaw fix,
     centred in x/y, lowest point at z = 0; head toward -Y = glTF +Z)
  2. swap in <work>/<critter>/tex_lifted.png as the texture, if present
  3. build the armature from profiles.json joints and the body-plan template
     (critter_templates.py); every bone's roll puts its local X on world +X,
     so a pitch is the same rotation on every bone
  4. skin: Blender's heat weights on a WELDED COPY (the glTF import splits
     vertices along UV seams, which heat diffusion reads as holes), copied
     back to the untouched mesh by nearest vertex; anything heat could not
     reach falls back to its nearest bone; normalised, at most 4 influences
  5. author the clips procedurally, keyed on every frame (quaternions, plus
     root location for rolls and hops): idle, walk, death for everyone,
     idle_flap for birds, idle_croak for the frog (the frog's walk IS a hop)
  6. save <out>/<critter>.blend and export <out>/character.glb (skinned mesh,
     deform bones, all actions, sampled, slid to zero, JPEG texture)

Clip names follow the Shards of Stone runtime's clip aliases: 'idle',
'walk', 'death'; 'idle_flap' and 'idle_croak' join the idle pool and play as
random idle variants. Loops close exactly (last key = first key).

Traced plans (critter_templates.MULTILEG: arachnid, crustacean, scorpion,
drake, sprawl) differ in three places: the joints come from fit_arthropod.py
(run it first; it also writes per-vertex limb labels that confine the heat
weights to each limb), the clips come from multileg.py (world-space poses
with per-leg IK: idle, walk, attack, death, special_1/2, with event markers
such as 'hit' and 'release'), and <out>/clips.json records every clip's
seconds, loop flag, event seconds and, for walks, the stride per cycle.
Options: --input <glb> overrides the profile's source; --set k=v[,k=v]
overrides `anim` values for a variant (e.g. --set heading_deg=0,face_yaw_deg=0
for a forward-walking crab from the sideways critter crab's profile).
"""
import bpy, bmesh, sys, os, json, math
import numpy as np
from mathutils import Vector, Quaternion, kdtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from critter_templates import bones_for, full_joints, MULTILEG  # noqa: E402
from paths import work_dir, source_glb  # noqa: E402
import multileg  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
NAME = opt['critter']
OUT = os.path.abspath(opt['out-dir'])
os.makedirs(OUT, exist_ok=True)
from paths import load_profiles  # noqa: E402
PROF = load_profiles()[NAME]
PLAN = PROF['plan']
ANIM = dict(PROF.get('anim', {}))
# --set key=value[,key=value]: per-run anim overrides for a variant (e.g. a
# forward-walking reef crab from the sideways critter crab's profile)
for kv in [x for x in opt.get('set', '').split(',') if x]:
    k, v = kv.split('=', 1)
    try:
        ANIM[k] = json.loads(v)
    except ValueError:
        ANIM[k] = v
FPS = 30
X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print('RIG', s)


# ── 1. import and normalise ────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
SRC = os.path.expanduser(opt['input']) if 'input' in opt else source_glb(NAME, PROF)
bpy.ops.import_scene.gltf(filepath=SRC)
mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
for o in bpy.context.scene.objects:
    o.select_set(o == mesh)
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(bpy.context.scene.objects):
    if o != mesh:
        bpy.data.objects.remove(o, do_unlink=True)
mesh.name = f'{NAME}_body'
me = mesh.data
co = np.empty(len(me.vertices) * 3, dtype=np.float32)
me.vertices.foreach_get('co', co)
co = co.reshape(-1, 3)
yaw = float(PROF.get('yaw_deg', 0.0))
pitch = float(PROF.get('pitch_deg', 0.0))
from rig_space import rig_rotation  # noqa: E402
R = rig_rotation(yaw, pitch)
co = co @ R
if yaw or pitch:
    # Rotate the imported custom split normals with the vertices: clearing
    # them instead would show every UV seam (the import splits vertices
    # there) as a shading crease.
    cn = np.empty(len(me.loops) * 3, dtype=np.float32)
    me.corner_normals.foreach_get('vector', cn)
    YAW_NORMALS = (cn.reshape(-1, 3) @ R).tolist()
else:
    YAW_NORMALS = None
# the SAME offset prep used (its views.json), so fit labels and joints line
# up vertex for vertex; bbox centring if there is no prep
vj = work_dir(NAME, 'prep', 'views.json')
if os.path.exists(vj):
    co += np.array(json.load(open(vj))['offset'], dtype=np.float32)
else:
    mn, mx = co.min(0), co.max(0)
    co += np.array([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]], dtype=np.float32)
me.vertices.foreach_set('co', co.ravel())
me.update()
if YAW_NORMALS is not None:
    me.normals_split_custom_set(YAW_NORMALS)
H = float(co[:, 2].max())
log(f'{NAME}: {len(me.vertices)} verts, height {H:.3f}, yaw {yaw}, pitch {pitch}')

# ── 2. texture ─────────────────────────────────────────────────────────────
lifted = work_dir(NAME, 'tex_lifted.png')
imgs = [i for i in bpy.data.images if i.size[0] > 0]
if os.path.exists(lifted) and imgs:
    img = imgs[0]
    img.filepath = lifted
    img.source = 'FILE'
    img.reload()
    img.pack()
    log('texture: levels-lifted', os.path.basename(lifted), tuple(img.size))
else:
    log('texture: original Meshy texture (no lifted file)')

# ── 3. armature ────────────────────────────────────────────────────────────
J = {k: Vector(v) for k, v in full_joints(PROF['joints']).items()}
BONES = bones_for(PLAN, J)
arm = bpy.data.armatures.new(f'{NAME}_rig')
rig = bpy.data.objects.new(f'{NAME}_rig', arm)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
for bn, h, t, parent in BONES:
    eb = arm.edit_bones.new(bn)
    eb.head, eb.tail = J[h], J[t]
    yax = (eb.tail - eb.head).normalized()
    ref = X if abs(yax.x) < 0.8 else Y
    xp = (ref - ref.dot(yax) * yax).normalized()
    eb.align_roll(xp.cross(yax))
    eb.use_deform = bn != 'root'
for bn, h, t, parent in BONES:
    if parent:
        arm.edit_bones[bn].parent = arm.edit_bones[parent]
bpy.ops.object.mode_set(mode='OBJECT')
NAMES = [b[0] for b in BONES]
log(f'armature: {len(NAMES)} bones ({PLAN})')

# ── 4. skin weights ────────────────────────────────────────────────────────
dup = mesh.copy()
dup.data = mesh.data.copy()
bpy.context.scene.collection.objects.link(dup)
bm = bmesh.new(); bm.from_mesh(dup.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
bm.to_mesh(dup.data); bm.free()
for o in bpy.context.scene.objects:
    o.select_set(o in (dup, rig))
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
deform = [n for n in NAMES if n != 'root']
kd = kdtree.KDTree(len(dup.data.vertices))
for v in dup.data.vertices:
    kd.insert(v.co, v.index)
kd.balance()
dgroups = {g.index: g.name for g in dup.vertex_groups}
dw = [dict() for _ in dup.data.vertices]
for v in dup.data.vertices:
    for g in v.groups:
        if g.weight > 1e-4 and dgroups[g.group] in deform:
            dw[v.index][dgroups[g.group]] = g.weight
bpy.data.objects.remove(dup, do_unlink=True)


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (a + ab * t - p).length


segs = {bn: (J[h], J[t]) for bn, h, t, _ in BONES if bn != 'root'}
groups = {n: mesh.vertex_groups.new(name=n) for n in deform}
fallback = 0
LABELS = None
if PLAN in MULTILEG:
    # fit_arthropod.py's per-vertex limb ownership (same vertex order: the
    # prep import and this one read the same file the same way)
    lp = work_dir(NAME, 'prep', 'labels.npy')
    LABELS = np.load(lp)
    LNAMES = json.load(open(work_dir(NAME, 'prep', 'labels.json')))['names']
    assert len(LABELS) == len(me.vertices), f'labels {len(LABELS)} != verts {len(me.vertices)}: re-run prep + fit'
    ALLOW = {i: multileg.allowed_bones(n, deform) & set(deform) for i, n in enumerate(LNAMES)}
    ALLOW[-1] = multileg.allowed_bones(None, deform) & set(deform)
    confined = 0
for v in me.vertices:
    _, idx, _ = kd.find(v.co)
    w = dict(dw[idx])
    if LABELS is not None:
        ok = ALLOW[int(LABELS[v.index])]
        w2 = {n: x for n, x in w.items() if n in ok}
        if sum(w2.values()) < 0.5 * sum(w.values()):
            confined += 1
        w = w2
        if sum(w.values()) < 1e-3:
            best = min(ok, key=lambda n: seg_dist(v.co, *segs[n]))
            w = {best: 1.0}
            fallback += 1
    if sum(w.values()) < 1e-3:
        # Heat could not reach this vertex (a detached island such as a
        # whisker or a bell): give it wholly to the nearest bone.
        best = min(segs, key=lambda n: seg_dist(v.co, *segs[n]))
        w = {best: 1.0}
        fallback += 1
    top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
    s = sum(x for _, x in top)
    for n, x in top:
        if x / s > 1e-3:
            groups[n].add([v.index], x / s, 'REPLACE')
if PLAN == 'bird':
    # Foot pads: every vertex at toe height near a foot belongs WHOLLY to that
    # foot's toe bone. Heat weights give the inner and outer toes to the shank,
    # so they tilted with it and dug into the ground whenever the shank did.
    pads = 0
    for sd in ('L', 'R'):
        fj = J[f'foot.{sd}']
        for v in me.vertices:
            if v.co.z <= fj.z + 0.01 and (v.co.xy - fj.xy).length < 0.45 and (v.co.x > 0) == (fj.x > 0):
                for n in deform:
                    groups[n].remove([v.index])
                groups[f'toe.{sd}'].add([v.index], 1.0, 'REPLACE')
                pads += 1
    log(f'foot pads: {pads} vertices given wholly to toe.L/toe.R')
counts = {n: 0 for n in deform}
for v in me.vertices:
    for g in v.groups:
        counts[mesh.vertex_groups[g.group].name] += 1
log(f'weights: heat on welded copy ({len(dw)} verts), {fallback} fallback-to-nearest')
if LABELS is not None:
    log(f'weights: confined to fit_arthropod limb labels; {confined} vertices lost over half their heat weight to it')
empty = [n for n, c in counts.items() if c == 0]
if empty:
    log('WARNING bones with no vertices:', empty)
mesh.parent = rig
mod = mesh.modifiers.new('Armature', 'ARMATURE')
mod.object = rig

# ── 5. clips ───────────────────────────────────────────────────────────────
REST = {b.name: b.matrix_local.to_quaternion() for b in arm.bones}
PB = rig.pose.bones
for pb in PB:
    pb.rotation_mode = 'QUATERNION'


def wq(bn, axis, deg):
    """Local rotation that turns bone `bn` by `deg` about a WORLD axis."""
    r = REST[bn]
    return r.inverted() @ Quaternion(axis, math.radians(deg)) @ r


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def bump(t, a, b, ramp=0.25):
    """0 outside [a, b], eased up over the first `ramp` of it, down over the last."""
    if t <= a or t >= b:
        return 0.0
    u = (t - a) / (b - a)
    return smooth(u / ramp) if u < ramp else (smooth((1 - u) / ramp) if u > 1 - ramp else 1.0)


def has(bn):
    return bn in PB


class Pose:
    def __init__(self):
        self.q = {}
        self.root_loc = Vector((0, 0, 0))
        self.scale = {}

    def rot(self, bn, axis, deg):
        if has(bn) and deg:
            self.q[bn] = wq(bn, axis, deg) @ self.q.get(bn, Quaternion())


def tail_chain(prefix='tail_'):
    out = []
    i = 1
    while has(f'{prefix}{i}'):
        out.append(f'{prefix}{i}')
        i += 1
    return out


TAIL = tail_chain()
BODY_W = float(np.percentile(np.abs(co[:, 0] - (J['spine'].x if 'spine' in J else 0.0)), 97))


# Quadruped legs: (upper, lower, foot, phase) in a lateral-sequence walk
# (LH, LF, RH, RF a quarter-cycle apart), as every four-legged walk is.
QLEGS = [('thigh.L', 'shin.L', 'foot.L', 0.0, 'hind'), ('upperarm.L', 'forearm.L', 'hand.L', 0.25, 'front'),
         ('thigh.R', 'shin.R', 'foot.R', 0.5, 'hind'), ('upperarm.R', 'forearm.R', 'hand.R', 0.75, 'front')]


def quad_walk(t, p):
    """Lateral-sequence walk. Each leg segment's WORLD pitch is authored and
    the local rotations derived from it, so a hoof held flat in stance really
    is flat (a local counter-rotation that ignores the chain above it tips the
    hoof into the ground). Stance: the leg is straight and sweeps back, hoof
    at its rest angle. Swing: the lower leg folds back (front: carpus; hind:
    stifle back, hock forward) and the leg comes forward."""
    A = ANIM.get('stride_deg', 22)
    F = ANIM.get('flex_deg', 40)
    D = 0.62
    for up, lo, ft, ph, kind in QLEGS:
        c = (t + ph) % 1.0
        if c < D:
            s = c / D
            w_up = -A + 2 * A * s          # touchdown forward -> lift-off back
            w_lo, w_ft = w_up, 0.0
        else:
            s = (c - D) / (1 - D)
            w_up = A - 2 * A * smooth(s)
            lift = math.sin(math.pi * s)
            if kind == 'front':
                w_lo = w_up + 1.35 * F * lift
                w_ft = 0.5 * w_lo              # hoof tilts back half as far: toe stays up
            else:
                w_lo = w_up + 0.8 * F * lift
                w_ft = w_lo - 1.1 * F * lift
        p.rot(up, X, w_up)
        p.rot(lo, X, w_lo - w_up)
        p.rot(ft, X, w_ft - w_lo)
    w = 2 * math.pi * t
    p.rot('chest', X, 1.5 * math.sin(2 * w))
    roll = ANIM.get('hip_roll_deg', 2.5) * math.sin(w)
    p.rot('hips', Y, roll)
    if TAIL:
        p.rot(TAIL[0], Y, -roll)          # a tail lying on the ground stays on it
    nod = ANIM.get('nod_deg', 4)
    p.rot('neck', X, nod * math.sin(2 * w + 0.6))
    p.rot('head', X, -0.5 * nod * math.sin(2 * w + 0.6))
    sway = ANIM.get('tail_sway_deg', 12)
    for i, bn in enumerate(TAIL):
        p.rot(bn, Z, sway * (0.6 + 0.25 * i) * math.sin(w - 0.7 * (i + 1)))


def quad_idle(t, p):
    style = ANIM.get('idle', 'graze')
    p.rot('chest', X, 1.2 * math.sin(2 * math.pi * 2 * t))       # two breaths
    look = 16 * bump(t, 0.02, 0.30) - 12 * bump(t, 0.72, 0.97)
    p.rot('neck', Z, look * 0.6)
    p.rot('head', Z, look * 0.4)
    if style == 'graze':
        g = bump(t, 0.30, 0.70, 0.3)
        p.rot('neck', X, ANIM.get('graze_deg', 34) * g)
        p.rot('head', X, (12 + 3 * math.sin(2 * math.pi * 10 * t)) * g)   # chewing
    elif style == 'sniff':
        p.rot('head', X, 4 * math.sin(2 * math.pi * 15 * t) * bump(t, 0.3, 0.7))
        p.rot('neck', X, 8 * bump(t, 0.3, 0.7))
    elif style == 'tilt':
        p.rot('head', Y, 12 * bump(t, 0.35, 0.65))
    wag = ANIM.get('idle_tail', 'sway')
    for i, bn in enumerate(TAIL):
        if wag == 'wag':
            p.rot(bn, Z, (18 + 6 * i) * math.sin(2 * math.pi * 9 * t - 0.5 * i))
        else:
            p.rot(bn, Z, (6 + 3 * i) * math.sin(2 * math.pi * 2 * t - 0.6 * i))


def death_roll(t, p, roll=90.0):
    """Keel over onto the left side (+X), pivoting on the ground, then settle.
    The root lifts by the body's half-width as it rolls so the flank lands on
    the ground instead of through it."""
    a = roll * (smooth((t - 0.12) / 0.5) + 0.07 * math.sin(math.pi * smooth((t - 0.62) / 0.3)))
    p.rot('root', Y, a)
    p.root_loc = Vector((0.0, 0.0, BODY_W * math.sin(math.radians(min(a, 90.0))) * 0.92))
    flinch = bump(t, 0.0, 0.22, 0.4)
    stiff = smooth((t - 0.2) / 0.5)
    p.rot('neck', X, -10 * flinch + 18 * stiff)
    p.rot('head', X, -8 * flinch + 10 * stiff)
    if ANIM.get('death_tail_counter') and TAIL:
        # A long tail lying on the ground (the rat's) keeps lying there
        # instead of swinging up into the air with the rolling body.
        p.rot(TAIL[0], Y, -a)
    else:
        for bn in TAIL:
            p.rot(bn, X, -20 * stiff)
    return stiff


def quad_death(t, p):
    stiff = death_roll(t, p)
    kick = 6 * math.sin(math.pi * 3 * smooth((t - 0.62) / 0.3)) * bump(t, 0.6, 0.95)
    for up, lo, ft, ph, kind in QLEGS:
        spread = -18 if kind == 'front' else 18
        p.rot(up, X, spread * stiff + kick)
        p.rot(lo, X, 8 * stiff)


BLEGS = [('thigh.L', 'shank.L', 'toe.L', 0.0), ('thigh.R', 'shank.R', 'toe.R', 0.5)]


def _ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def ik2(hip, knee, foot, target):
    """Analytic two-bone IK in the sagittal (y, z) plane. Returns the new knee
    position, bending to the same side as the rest pose."""
    L1 = (knee - hip).length; L2 = (foot - knee).length
    d = target - hip
    dist = max(abs(L1 - L2) + 1e-4, min(L1 + L2 - 1e-4, d.length))
    ca = max(-1.0, min(1.0, (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)))
    a = math.acos(ca)
    r0, k0 = foot - hip, knee - hip
    side = 1.0 if (r0.x * k0.y - r0.y * k0.x) > 0 else -1.0
    base = math.atan2(d.y, d.x)
    ang = base + side * a
    return hip + Vector((math.cos(ang), math.sin(ang))) * L1


def bird_walk(t, p):
    """Two-legged walk driven by a FOOT PATH, solved with two-bone IK: in
    stance the foot sits on the ground sliding back, in swing it lifts on an
    arc and comes forward. The toes stay flat in stance and tip up in swing.
    Bird shanks rest angled forward, so rotating them by hand dips the foot
    before it lifts; IK cannot put the foot anywhere but on its path."""
    D = 0.56
    for th, sh, toe, ph in BLEGS:
        sd = th[-1]
        hip = J[f'hip.{sd}'].yz; knee = J[f'ankle.{sd}'].yz; foot = J[f'foot.{sd}'].yz
        hh = hip.y  # hip height (z) in the (y, z) plane
        S = ANIM.get('stride', 0.55) * hh
        lift_h = ANIM.get('step_height', 0.22) * hh
        c = (t + ph) % 1.0
        if c < D:
            s = c / D
            tgt = foot + Vector((-S / 2 + S * s, 0.0))
            w_toe = 0.0
        else:
            s = (c - D) / (1 - D)
            lift = math.sin(math.pi * s)
            tgt = foot + Vector((S / 2 - S * smooth(s), lift_h * lift))
            w_toe = -20 * lift
        kn = ik2(hip, knee, foot, tgt)
        w_th = _ang(kn - hip) - _ang(knee - hip)
        w_sh = _ang(tgt - kn) - _ang(foot - knee)
        p.rot(th, X, w_th)
        p.rot(sh, X, w_sh - w_th)
        p.rot(toe, X, w_toe - w_sh)
    w = 2 * math.pi * t
    bob = ANIM.get('head_bob_deg', 10)
    p.rot('neck', X, bob * math.sin(2 * w))
    p.rot('head', X, -bob * math.sin(2 * w))
    waddle = ANIM.get('waddle_deg', 4) * math.sin(w)
    p.rot('hips', Y, waddle)
    for th, sh, toe, ph in BLEGS:
        p.rot(th, Y, -waddle)             # the body rolls, the legs stay upright
    for i, bn in enumerate(TAIL):
        p.rot(bn, X, 4 * math.sin(2 * w - 0.5 * (i + 1)))


def bird_idle(t, p):
    p.rot('chest', X, 1.5 * math.sin(2 * math.pi * 2 * t))
    look = 22 * bump(t, 0.02, 0.28) - 18 * bump(t, 0.74, 0.98)
    p.rot('neck', Z, look * 0.5)
    p.rot('head', Z, look * 0.5)
    if ANIM.get('idle', 'peck') == 'peck':
        peck = max(bump(t, 0.36, 0.48, 0.45), bump(t, 0.52, 0.64, 0.45))
        p.rot('neck', X, ANIM.get('peck_deg', 48) * peck)
        p.rot('head', X, 18 * peck)
    else:  # a crow tilts its head at you
        p.rot('head', Y, 16 * bump(t, 0.34, 0.66))
    for i, bn in enumerate(TAIL):
        p.rot(bn, X, 5 * math.sin(2 * math.pi * 3 * t - 0.5 * i))


def bird_flap(t, p):
    """Wings open off the flanks and flutter (3 beats), body lifts a little."""
    env = bump(t, 0.02, 0.98, 0.18)
    beat = 0.5 - 0.5 * math.cos(2 * math.pi * 3 * t)
    open_ = 1.15 * ANIM.get('flap_deg', 62) * env * (0.35 + 0.65 * beat)
    for side, sg in (('L', 1), ('R', -1)):
        p.rot(f'wing.{side}', Y, -sg * open_)
        p.rot(f'wing.{side}', X, -12 * env)
        p.rot(f'wingtip.{side}', Y, -sg * 0.35 * open_ * (0.5 + 0.5 * math.cos(2 * math.pi * 3 * t - 0.8)))
    p.rot('chest', X, -8 * env)
    p.rot('neck', X, -10 * env)
    for bn in TAIL:
        p.rot(bn, X, 10 * env)


def bird_death(t, p):
    stiff = death_roll(t, p)
    for th, sh, toe, ph in BLEGS:
        p.rot(th, X, -25 * stiff)
        p.rot(toe, X, 35 * stiff)
    for side, sg in (('L', 1), ('R', -1)):
        p.rot(f'wing.{side}', Y, -sg * 25 * stiff)


FLEGS = [('thigh.L', 'shin.L', 'foot.L'), ('thigh.R', 'shin.R', 'foot.R')]
FARMS = [('upperarm.L', 'forearm.L', 'hand.L'), ('upperarm.R', 'forearm.R', 'hand.R')]


def _yaw_deg(v):
    return math.degrees(math.atan2(v.y, v.x))


def frog_hop(t, p):
    """Crouch, push off, fly with the legs trailing, land on the hands, fold.
    A sitting frog's hind leg is folded FLAT in the horizontal plane (thigh
    forward-out, shin back, foot forward-out), so the jump unfolds it by YAW:
    the thigh swings back-out, the shin keeps pointing back and the foot
    swings back, leaving the leg straight behind. (Pitching the forward
    thigh instead drives the knee straight through the floor.)"""
    # `hop_unfold` caps the unfold: a frog's thighs ARE its big rounded rear
    # (heat gives them a third of the mesh), so a full unfold drags the body
    # into a long stretched shape. A partial kick keeps the silhouette.
    ext = ANIM.get('hop_unfold', 1.0) * smooth((t - 0.16) / 0.14) * (1 - smooth((t - 0.56) / 0.22))
    u = max(0.0, min(1.0, (t - 0.2) / 0.55))
    air = math.sin(math.pi * u)
    p.root_loc = Vector((0, 0, ANIM.get('hop_height', 0.3) * H * air))
    # Nose up while rising, nose down while falling, each scaled by the
    # height already gained, so the tilt never swings the trailing legs or
    # the reaching hands below the ground the body has left.
    p.rot('root', X, (-14 if u < 0.5 else 10) * air)
    for sd, sg in (('L', 1), ('R', -1)):
        hip, knee, ankle, toe = (J[f'{j}.{sd}'] for j in ('hip', 'knee', 'ankle', 'toe'))
        th0, sh0, ft0 = _yaw_deg(knee - hip), _yaw_deg(ankle - knee), _yaw_deg(toe - ankle)
        # flight pose: every segment pointing straight back-and-out
        th1, sh1, ft1 = sg * 72.0 + (0 if sg > 0 else 180), sg * 84.0 + (0 if sg > 0 else 180), sg * 80.0 + (0 if sg > 0 else 180)

        def turn(a0, a1):
            d = (a1 - a0 + 180.0) % 360.0 - 180.0
            if sg > 0 and d < 0:
                d += 360.0
            if sg < 0 and d > 0:
                d -= 360.0
            return d
        w_th = turn(th0, th1) * ext
        w_sh = turn(sh0, sh1) * ext
        w_ft = turn(ft0, ft1) * ext
        p.rot(f'thigh.{sd}', Z, w_th)
        p.rot(f'shin.{sd}', Z, w_sh - w_th)
        p.rot(f'foot.{sd}', Z, w_ft - w_sh)
    for ua, fa, hd in FARMS:
        p.rot(ua, X, -28 * air + 14 * bump(t, 0.66, 0.9, 0.5))
        p.rot(fa, X, 12 * air)


def frog_idle(t, p):
    breath = 0.5 - 0.5 * math.cos(2 * math.pi * 2 * t)
    p.scale['throat'] = 1.0 + 0.08 * breath
    look = 10 * bump(t, 0.05, 0.35) - 8 * bump(t, 0.62, 0.92)
    p.rot('head', Z, look)
    p.rot('chest', X, 1.0 * math.sin(2 * math.pi * 2 * t))


def frog_croak(t, p):
    puff = max(bump(t, 0.05, 0.45, 0.4), bump(t, 0.52, 0.95, 0.4))
    p.scale['throat'] = 1.0 + ANIM.get('croak_scale', 0.42) * puff
    p.rot('head', X, -6 * puff)


def frog_death(t, p):
    stiff = death_roll(t, p)
    for th, sh, ft in FLEGS:
        p.rot(th, X, 30 * stiff)
        p.rot(sh, X, -25 * stiff)
    for ua, fa, hd in FARMS:
        p.rot(ua, X, -20 * stiff)


CLIPS = multileg.clips_for(PLAN, ANIM) if PLAN in MULTILEG else {
    'quadruped': [('idle', 90, quad_idle, True), ('walk', ANIM.get('walk_frames', 30), quad_walk, True),
                  ('death', 36, quad_death, False)],
    'bird': [('idle', 90, bird_idle, True), ('idle_flap', 40, bird_flap, True),
             ('walk', ANIM.get('walk_frames', 20), bird_walk, True), ('death', 30, bird_death, False)],
    'frog': [('idle', 90, frog_idle, True), ('idle_croak', 45, frog_croak, True),
             ('walk', ANIM.get('walk_frames', 24), frog_hop, True), ('death', 30, frog_death, False)],
}[PLAN]
if PLAN not in MULTILEG:
    CLIPS = [c + ({},) for c in CLIPS]
MRIG = multileg.Rig(BONES, J, REST, ANIM, H) if PLAN in MULTILEG else None

scene = bpy.context.scene
scene.render.fps = FPS
rig.animation_data_create()
dg = bpy.context.evaluated_depsgraph_get()


def mesh_min_z():
    dg.update()
    ev = mesh.evaluated_get(dg)
    m = ev.to_mesh()
    arr = np.empty(len(m.vertices) * 3, dtype=np.float64)
    m.vertices.foreach_get('co', arr)
    ev.to_mesh_clear()
    return float(arr.reshape(-1, 3)[:, 2].min())


def ground_solve(action, frames, lift_only=False, loop=True):
    """Death ends lying ON the ground, not in or above it. The analytic
    roll-lift uses the body's half-width, but horns, ears, a bell or a beak
    stick out past it. Evaluate the deformed mesh on every frame and raise
    (or, once settled, lower) the root so its lowest point meets z = 0, from
    the moment the fall begins; smoothed so the correction never pops."""
    corr = []
    raw_mz = []
    for f in range(frames + 1):
        scene.frame_set(f + 1)
        t = f / frames
        mz = mesh_min_z()
        c = 0.0
        if lift_only:
            c = max(0.0, -mz)          # legs push the body up, never through the floor
        elif t >= 0.12:
            c = -mz if mz < 0 else (-mz if t >= 0.62 else 0.0)
        corr.append(c)
        raw_mz.append(mz)
    if lift_only and not loop:
        # One-shot (attack, cast): lift only, smoothed, ends free.
        sm = [max(corr[max(0, i - 2):i + 3]) for i in range(len(corr))]
        sm = [sum(sm[max(0, i - 1):i + 2]) / len(sm[max(0, i - 1):i + 2]) for i in range(len(sm))]
    elif lift_only:
        # Loop clip: the last key IS the first pose, so smooth cyclically and
        # pin the ends equal or the loop acquires a seam.
        corr[-1] = corr[0]
        n = len(corr) - 1
        # max over 5 then mean over 3: every frame is lifted at least as far
        # as it needs, and the curve stays smooth.
        mx = [max(corr[(i + k) % n] for k in range(-2, 3)) for i in range(n)]
        sm = [(mx[(i - 1) % n] + mx[i] + mx[(i + 1) % n]) / 3 for i in range(n)]
        sm.append(sm[0])
    else:
        sm = [sum(corr[max(0, i - 1):i + 2]) / len(corr[max(0, i - 1):i + 2]) for i in range(len(corr))]
    for f in range(frames + 1):
        if abs(sm[f]) < 1e-6:
            continue
        scene.frame_set(f + 1)
        world = root_rest @ PB['root'].location
        world = Vector((world.x, world.y, world.z + sm[f]))
        PB['root'].location = root_rest.inverted() @ world
        PB['root'].keyframe_insert('location', frame=f + 1, group='root')
    log(f'{action.name} ground solve: max lift {max(sm):.3f}, final {sm[-1]:+.3f}')
    log(f'{action.name} ground solve per-frame: ' + ' '.join(f'{x:.3f}' for x in sm))
    log(f'{action.name} raw min z per-frame: ' + ' '.join(f'{x:.3f}' for x in raw_mz))
root_rest = REST['root']
CLIP_TABLE = {}
for clip, frames, fn, loop, events in CLIPS:
    action = bpy.data.actions.new(clip)
    rig.animation_data.action = action
    clamped = 0
    clamp_legs = {}
    info = {}
    for f in range(frames + 1):
        t = f / frames
        if MRIG:
            wp = multileg.WPose(MRIG)
            info = fn(t if not loop or f < frames else 0.0, wp) or info
            p = Pose()
            p.q = wp.local()
            p.root_loc = wp.root_loc
            p.scale = wp.scale
            clamped += wp.clamped
            for k, v in wp.clamp_legs.items():
                clamp_legs[k] = max(clamp_legs.get(k, 0.0), v)
        else:
            p = Pose()
            fn(t if not loop or f < frames else 0.0, p)  # loops close exactly
        for bn in NAMES:
            pb = PB[bn]
            pb.rotation_quaternion = p.q.get(bn, Quaternion())
            pb.keyframe_insert('rotation_quaternion', frame=f + 1, group=bn)
            s = p.scale.get(bn, 1.0)
            pb.scale = (s, s, s)
            if bn in p.scale or (PLAN == 'frog' and bn == 'throat') or (MRIG and bn == 'abdomen'):
                pb.keyframe_insert('scale', frame=f + 1, group=bn)
        PB['root'].location = root_rest.inverted() @ p.root_loc
        PB['root'].keyframe_insert('location', frame=f + 1, group='root')
    if clip == 'death':
        ground_solve(action, frames)
    elif PLAN == 'frog' and clip == 'walk':
        ground_solve(action, frames, lift_only=True)
    elif MRIG and PLAN != 'drake':   # a flyer's clips never touch the ground
        # IK plants the foot TIP joint; a tarsus tilting about it can push
        # other foot vertices a hair under the ground. Lift, never sink.
        ground_solve(action, frames, lift_only=True, loop=loop)
    # Event markers (Action pose markers; seconds in clips.json): the hit of
    # an attack, the release of a cast. A frame number is not a timing.
    for ev, frac in events.items():
        action.pose_markers.new(ev).frame = 1 + int(round(frac * frames))
    CLIP_TABLE[clip] = {'frames': frames, 'seconds': round(frames / FPS, 3), 'loop': loop,
                        'events': {ev: round(frac * frames / FPS, 3) for ev, frac in events.items()}}
    CLIP_TABLE[clip].update({k: round(v, 4) for k, v in info.items()})
    if clamped:
        log(f'{clip}: {clamped} leg-frames had an unreachable foot target (clamped); worst shortfall per leg: '
            + ' '.join(f'{k}={v:.3f}' for k, v in sorted(clamp_legs.items())))
    track = rig.animation_data.nla_tracks.new()
    track.name = clip
    strip = track.strips.new(clip, 1, action)
    strip.action_slot = rig.animation_data.action_slot
    track.mute = True
    log(f'clip {clip}: {frames} frames = {frames / FPS:.2f} s, {"loop" if loop else "one-shot"}')
    # reset the pose between clips
    for pb in PB:
        pb.rotation_quaternion = Quaternion(); pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
json.dump({'critter': NAME, 'plan': PLAN, 'fps': FPS, 'source': SRC, 'clips': CLIP_TABLE},
          open(os.path.join(OUT, 'clips.json'), 'w'), indent=1)
first = CLIPS[0][0]
rig.animation_data.action = bpy.data.actions[first]
rig.animation_data.action_slot = bpy.data.actions[first].slots[0]
scene.frame_start, scene.frame_end = 1, CLIPS[0][1] + 1
scene.frame_set(1)

# ── 6. save and export ─────────────────────────────────────────────────────
FACE = float(ANIM.get('face_yaw_deg', 0.0))
if FACE:
    # The game moves a unit along its facing (glTF +Z = rig -Y). A crab that
    # scuttles sideways (walk heading_deg 90, toward +X) must present its
    # SIDE as its facing: turn rig and skin together about Z and apply it to
    # the data. Pose-local keys ride along (every bone's rest turns with it).
    for o in bpy.context.scene.objects:
        o.select_set(o in (mesh, rig))
    bpy.context.view_layer.objects.active = rig
    rig.rotation_euler = (0.0, 0.0, math.radians(FACE))
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    log(f'facing: rig and skin turned {FACE:+.0f} deg about Z (walk heading {ANIM.get("heading_deg", 0)})')
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f'{NAME}.blend'))
for o in bpy.context.scene.objects:
    o.select_set(o in (mesh, rig))
bpy.context.view_layer.objects.active = rig
bpy.ops.export_scene.gltf(
    filepath=os.path.join(OUT, 'character.glb'), export_format='GLB', use_selection=True,
    export_animations=True, export_animation_mode='ACTIONS', export_force_sampling=True,
    export_anim_slide_to_zero=True, export_skins=True, export_all_influences=False,
    export_image_format='JPEG', export_jpeg_quality=92, export_yup=True, export_apply=False)
json.dump({'critter': NAME, 'plan': PLAN, 'log': LOG, 'bone_vertex_counts': counts},
          open(os.path.join(OUT, 'rig-report.json'), 'w'), indent=1)
log('exported', os.path.join(OUT, 'character.glb'))
