# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Bone templates per body plan, shared by the fitting jig and the rig script.

A bone is (name, head_joint, tail_joint, parent). Joints live in the
critter's profile (profiles.json `joints`); a bone whose joints are missing is
simply skipped, so a critter can omit optional parts (a frog has no tail).
Tail chains are variable length: joints tail_0 .. tail_N give bones
tail_1 .. tail_N. Deform bones only: nothing here is a control or IK target,
so the exported GLB carries exactly these bones.

Axes (rig space): head toward -Y, left = +X, up = +Z. Every bone gets its
roll aligned so its local X axis points to world +X (see rig_critter.py), so
"pitch" (rotation about local X) means the same thing on every bone.
"""

SIDES = ('L', 'R')


def _tail(joints, parent):
    n = 0
    while f'tail_{n + 1}' in joints:
        n += 1
    out = []
    prev = parent
    for i in range(1, n + 1):
        out.append((f'tail_{i}', f'tail_{i - 1}', f'tail_{i}', prev))
        prev = f'tail_{i}'
    return out


def quadruped(joints):
    b = [
        ('root', 'root', 'root_tip', None),
        ('hips', 'hips', 'spine', 'root'),
        ('chest', 'spine', 'chest', 'hips'),
        ('neck', 'neck_base', 'head_base', 'chest'),
        ('head', 'head_base', 'head_tip', 'neck'),
    ]
    for s in SIDES:
        b += [
            (f'upperarm.{s}', f'shoulder.{s}', f'elbow.{s}', 'chest'),
            (f'forearm.{s}', f'elbow.{s}', f'fetlock.{s}', f'upperarm.{s}'),
            (f'hand.{s}', f'fetlock.{s}', f'ftoe.{s}', f'forearm.{s}'),
            (f'thigh.{s}', f'hip.{s}', f'stifle.{s}', 'hips'),
            (f'shin.{s}', f'stifle.{s}', f'hock.{s}', f'thigh.{s}'),
            (f'foot.{s}', f'hock.{s}', f'htoe.{s}', f'shin.{s}'),
        ]
    b += _tail(joints, 'hips')
    return b


def bird(joints):
    b = [
        ('root', 'root', 'root_tip', None),
        ('hips', 'hips', 'spine', 'root'),
        ('chest', 'spine', 'chest', 'hips'),
        ('neck', 'neck_base', 'head_base', 'chest'),
        ('head', 'head_base', 'head_tip', 'neck'),
    ]
    for s in SIDES:
        b += [
            (f'thigh.{s}', f'hip.{s}', f'ankle.{s}', 'hips'),
            (f'shank.{s}', f'ankle.{s}', f'foot.{s}', f'thigh.{s}'),
            (f'toe.{s}', f'foot.{s}', f'toe.{s}', f'shank.{s}'),
            (f'wing.{s}', f'wing_root.{s}', f'wing_mid.{s}', 'chest'),
            (f'wingtip.{s}', f'wing_mid.{s}', f'wing_tip.{s}', f'wing.{s}'),
        ]
    b += _tail(joints, 'hips')
    return b


def frog(joints):
    b = [
        ('root', 'root', 'root_tip', None),
        ('hips', 'hips', 'spine', 'root'),
        ('chest', 'spine', 'chest', 'hips'),
        ('head', 'head_base', 'head_tip', 'chest'),
        ('throat', 'throat_base', 'throat_tip', 'chest'),
    ]
    for s in SIDES:
        b += [
            (f'upperarm.{s}', f'shoulder.{s}', f'elbow.{s}', 'chest'),
            (f'forearm.{s}', f'elbow.{s}', f'wrist.{s}', f'upperarm.{s}'),
            (f'hand.{s}', f'wrist.{s}', f'finger.{s}', f'forearm.{s}'),
            (f'thigh.{s}', f'hip.{s}', f'knee.{s}', 'hips'),
            (f'shin.{s}', f'knee.{s}', f'ankle.{s}', f'thigh.{s}'),
            (f'foot.{s}', f'ankle.{s}', f'toe.{s}', f'shin.{s}'),
        ]
    return b


LEG_SEGS = ('coxa', 'femur', 'tibia', 'tarsus')


def leg_count(joints, side='L'):
    n = 0
    while f'leg{n + 1}_0.{side}' in joints:
        n += 1
    return n


def _legs(joints, parent='body'):
    """Walking legs leg1 (front) .. legN (back) per side, 4 bones each, on
    joints leg<i>_0 (root, inside the body) .. leg<i>_4 (foot tip).
    `parent` is a bone name or a function of the leg number."""
    out = []
    for s in SIDES:
        for i in range(1, leg_count(joints, s) + 1):
            prev = parent(i) if callable(parent) else parent
            for k, seg in enumerate(LEG_SEGS):
                bn = f'leg{i}_{seg}.{s}'
                out.append((bn, f'leg{i}_{k}.{s}', f'leg{i}_{k + 1}.{s}', prev))
                prev = bn
    return out


def _claws(parent='body'):
    """A chela: arm (merus), forearm (carpus), palm (propodus, which carries
    the FIXED finger to its tip) and the moving finger (dactyl), hinged at
    the gape. The pincer opens by turning claw_finger alone."""
    b = []
    for s in SIDES:
        b += [
            (f'claw_arm.{s}', f'claw_0.{s}', f'claw_1.{s}', parent),
            (f'claw_fore.{s}', f'claw_1.{s}', f'claw_2.{s}', f'claw_arm.{s}'),
            (f'claw_palm.{s}', f'claw_2.{s}', f'claw_4.{s}', f'claw_fore.{s}'),
            (f'claw_finger.{s}', f'dactyl_0.{s}', f'dactyl_1.{s}', f'claw_palm.{s}'),
        ]
    return b


def _pairs(kind, parent='body', segs=1):
    b = []
    for s in SIDES:
        prev = parent
        for k in range(segs):
            bn = f'{kind}.{s}' if k == 0 else f'{kind}{k + 1}.{s}'
            b.append((bn, f'{kind}_{k}.{s}', f'{kind}_{k + 1}.{s}', prev))
            prev = bn
    return b


def _arthropod_core():
    return [
        ('root', 'root', 'root_tip', None),
        ('body', 'body_0', 'body_1', 'root'),
    ]


def arachnid(joints):
    """Spider: cephalothorax (`body`), abdomen hinged at the pedicel, a
    spinneret, two fangs (chelicerae), optional pedipalps, N legs a side."""
    b = _arthropod_core() + [
        ('abdomen', 'abdomen_0', 'abdomen_1', 'body'),
        ('spinneret', 'abdomen_1', 'spinneret_1', 'abdomen'),
    ]
    b += _pairs('fang') + _pairs('palp', segs=2)
    b += _legs(joints)
    return b


def crustacean(joints):
    """Crab: one carapace bone, two chelae with an opening pincer, two eye
    stalks, N walking legs a side."""
    return _arthropod_core() + _claws() + _pairs('eye') + _legs(joints)


def scorpion(joints):
    """Scorpion: prosoma (`body`, carries legs, pedipalp claws and
    chelicerae), mesosoma (`abdomen`), and the metasoma as tail_1..tail_N,
    the last segment being the telson with the sting."""
    b = _arthropod_core() + [('abdomen', 'abdomen_0', 'tail_0', 'body')]
    b += _claws() + _pairs('fang') + _legs(joints)
    b += _tail(joints, 'abdomen')
    return b


def _count(joints, prefix):
    n = 0
    while f'{prefix}_{n + 1}' in joints:
        n += 1
    return n


def _vertebrate(joints, wings=False):
    """Spine spine1 (hips) .. spineN (chest) on joints spine_0 .. spine_N,
    neck1..neckM + head on neck_0 .. neck_M, head_tip; the tail hangs off
    the hips; front legs (leg1) off the chest, hind legs off the hips, each
    the same 4-bone chain the arthropod legs use (coxa = the girdle's short
    link inside the body), so the same IK and gait solve them."""
    ns = _count(joints, 'spine')
    b = [('root', 'root', 'root_tip', None)]
    prev = 'root'
    for k in range(1, ns + 1):
        b.append((f'spine{k}', f'spine_{k - 1}', f'spine_{k}', prev))
        prev = f'spine{k}'
    chest = prev
    nn = _count(joints, 'neck')
    for k in range(1, nn + 1):
        b.append((f'neck{k}', f'neck_{k - 1}', f'neck_{k}', prev))
        prev = f'neck{k}'
    b.append(('head', f'neck_{nn}', 'head_tip', prev))
    b += _tail(joints, 'spine1')
    # front legs off the chest, the rest off the hips; a wyvern's only pair
    # (its wings are its arms) is a HIND pair
    nl = leg_count(joints)
    b += _legs(joints, lambda i: chest if (i == 1 and nl > 1) else 'spine1')
    if wings:
        for s in SIDES:
            b += [(f'wing_arm.{s}', f'wing_0.{s}', f'wing_1.{s}', chest),
                  (f'wing_fore.{s}', f'wing_1.{s}', f'wing_2.{s}', f'wing_arm.{s}'),
                  (f'wing_hand.{s}', f'wing_2.{s}', f'wing_3.{s}', f'wing_fore.{s}')]
    return b


def drake(joints):
    """Winged quadruped (the drakeling): vertebrate + a 3-bone wing chain
    a side (arm, forearm, hand) whose membrane follows by heat weights."""
    return _vertebrate(joints, wings=True)


def sprawl(joints):
    """Low sprawling quadruped (the magma salamander): legs out to the
    side, a long tail, a flexible spine for lateral undulation."""
    return _vertebrate(joints)


PLANS = {'quadruped': quadruped, 'bird': bird, 'frog': frog,
         'arachnid': arachnid, 'crustacean': crustacean, 'scorpion': scorpion,
         'drake': drake, 'sprawl': sprawl}
# plans fitted by fit_arthropod.py (traced limbs + labels) and posed by
# multileg.py (world-space poses, leg IK)
MULTILEG = ('arachnid', 'crustacean', 'scorpion', 'drake', 'sprawl')


def bones_for(plan, joints):
    """Template bones whose head and tail joints both exist."""
    allb = PLANS[plan](joints)
    parent = {bn[0]: bn[3] for bn in allb}
    keep = [bn for bn in allb if bn[1] in joints and bn[2] in joints]
    names = {bn[0] for bn in keep}
    out = []
    for n, h, t, p in keep:
        while p is not None and p not in names:   # a skipped bone's child
            p = parent.get(p)                     # hangs off its grandparent
        out.append((n, h, t, p))
    return out


def full_joints(joints):
    """Joints with '.R' filled from '.L' by mirroring x, ONLY where the
    profile gives no explicit '.R' (Meshy poses are often asymmetric)."""
    out = dict(joints)
    for k, v in joints.items():
        if k.endswith('.L') and k[:-2] + '.R' not in joints:
            out[k[:-2] + '.R'] = [-v[0], v[1], v[2]]
    return out
