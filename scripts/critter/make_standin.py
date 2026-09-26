# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Procedural stand-in meshes for body plans whose Meshy meshes don't exist yet.

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/critter/make_standin.py -- --kind spider --out <dir>/spider.glb

Kinds: spider (8 arched legs, fangs, palps, spinneret), scorpion (8 legs,
pedipalp claws with an open gape, a 5-segment tail curled over the back and
a sting), drake (quadruped, membrane wings, neck, tail), salamander (a low
sprawling quadruped with a long tail). Built with Blender's Skin modifier
over a skeleton of vertices and edges, then one level of subdivision: one
connected quad mesh with tube limbs, like a clean sculpt, not a primitive
pile. Rig space, as the critter tools expect: head toward -Y, left = +X,
feet at z = 0. The stand-in proves a template and its clips; it is not art.
"""
import bpy, sys, math, os

argv = sys.argv[sys.argv.index('--') + 1:]
opt = {argv[i][2:]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
KIND = opt['kind']


class Skel:
    def __init__(self):
        self.v, self.r, self.e = [], [], []

    def add(self, p, r, link=None):
        self.v.append(p); self.r.append(r)
        i = len(self.v) - 1
        if link is not None:
            self.e.append((link, i))
        return i

    def chain(self, start, pts):
        prev = start
        out = []
        for p, r in pts:
            prev = self.add(p, r, prev)
            out.append(prev)
        return out


def mirror_x(p):
    return (-p[0], p[1], p[2])


def spider(S):
    spine = [((0, -0.40, 0.31), 0.06), ((0, -0.33, 0.33), 0.12), ((0, -0.22, 0.35), 0.16),
             ((0, -0.10, 0.35), 0.13), ((0, -0.02, 0.36), 0.05), ((0, 0.05, 0.38), 0.12),
             ((0, 0.17, 0.42), 0.24), ((0, 0.32, 0.43), 0.25), ((0, 0.45, 0.39), 0.16),
             ((0, 0.52, 0.35), 0.06)]
    ids = []
    for p, r in spine:
        ids.append(S.add(p, r, ids[-1] if ids else None))
    S.chain(ids[-1], [((0, 0.57, 0.33), 0.03)])                          # spinneret
    for sg in (1, -1):
        m = (lambda p: p) if sg > 0 else mirror_x
        for k, ang in enumerate((32, 68, 108, 146)):
            a = math.radians(ang)
            d = (math.sin(a), -math.cos(a))
            y0 = -0.30 + 0.07 * k
            base = S.add(m((0.10, y0, 0.33)), 0.045, ids[2] if k < 2 else ids[3])
            L = 1.0 if k in (0, 3) else 0.88

            def at(out, z):
                return m((0.10 + d[0] * out * L, y0 + d[1] * out * L, z))
            S.chain(base, [(at(0.08, 0.38), 0.04), (at(0.30, 0.60), 0.032), (at(0.55, 0.30), 0.025),
                           (at(0.66, 0.02), 0.012)])
        S.chain(ids[0], [(m((0.035, -0.43, 0.28)), 0.03), (m((0.04, -0.46, 0.20)), 0.02),
                         (m((0.035, -0.45, 0.14)), 0.008)])               # fang
        S.chain(ids[0], [(m((0.07, -0.43, 0.32)), 0.022), (m((0.11, -0.52, 0.30)), 0.018),
                         (m((0.12, -0.58, 0.20)), 0.012)])                # palp
    return (0.10, 0.07, 0.06), (0.35, 0.12, 0.08)


def scorpion(S):
    # one spine vertex per leg pair: the Skin modifier builds a hull at every
    # branch vertex, and 4+ branches on one vertex give a boxy flange
    spine = [((0, -0.44, 0.18), 0.05), ((0, -0.37, 0.19), 0.10), ((0, -0.26, 0.20), 0.13),
             ((0, -0.16, 0.21), 0.15), ((0, -0.06, 0.22), 0.16), ((0, 0.04, 0.22), 0.16),
             ((0, 0.14, 0.22), 0.15), ((0, 0.24, 0.23), 0.12), ((0, 0.34, 0.25), 0.08)]
    ids = []
    for p, r in spine:
        ids.append(S.add(p, r, ids[-1] if ids else None))
    S.chain(ids[-1], [((0, 0.42, 0.33), 0.07), ((0, 0.47, 0.47), 0.065), ((0, 0.47, 0.62), 0.06),
                      ((0, 0.41, 0.75), 0.06), ((0, 0.31, 0.83), 0.055), ((0, 0.20, 0.84), 0.075),
                      ((0, 0.12, 0.79), 0.05), ((0, 0.08, 0.70), 0.008)])   # tail + telson
    for sg in (1, -1):
        m = (lambda p: p) if sg > 0 else mirror_x
        for k, ang in enumerate((55, 80, 105, 130)):
            a = math.radians(ang)
            d = (math.sin(a), -math.cos(a))
            y0 = -0.26 + 0.10 * k
            base = S.add(m((0.11, y0, 0.19)), 0.04, ids[2 + k])

            def at(out, z):
                return m((0.11 + d[0] * out, y0 + d[1] * out, z))
            S.chain(base, [(at(0.07, 0.22), 0.035), (at(0.25, 0.33), 0.028), (at(0.42, 0.14), 0.02),
                           (at(0.50, 0.01), 0.01)])
        palm = S.chain(ids[1], [(m((0.12, -0.44, 0.20)), 0.045), (m((0.28, -0.52, 0.22)), 0.04),
                                (m((0.28, -0.66, 0.21)), 0.05), (m((0.27, -0.78, 0.21)), 0.08),
                                (m((0.26, -0.86, 0.21)), 0.06)])[-1]
        S.chain(palm, [(m((0.21, -0.95, 0.20)), 0.035), (m((0.19, -1.03, 0.20)), 0.01)])   # fixed finger
        S.chain(palm, [(m((0.31, -0.95, 0.21)), 0.035), (m((0.30, -1.02, 0.21)), 0.01)])   # dactyl
        S.chain(ids[0], [(m((0.03, -0.48, 0.15)), 0.022), (m((0.03, -0.51, 0.11)), 0.008)])   # chelicera
    return None, None


def drake(S):
    spine = [((0, 0.30, 0.52), 0.14), ((0, 0.12, 0.56), 0.19), ((0, -0.08, 0.58), 0.2),
             ((0, -0.24, 0.62), 0.15)]
    ids = []
    for p, r in spine:
        ids.append(S.add(p, r, ids[-1] if ids else None))
    neck = S.chain(ids[-1], [((0, -0.34, 0.74), 0.1), ((0, -0.42, 0.88), 0.08), ((0, -0.50, 0.96), 0.09),
                             ((0, -0.62, 0.96), 0.08), ((0, -0.74, 0.92), 0.05), ((0, -0.80, 0.90), 0.02)])
    for sg in (1, -1):
        m = (lambda p: p) if sg > 0 else mirror_x
        S.chain(neck[2], [(m((0.05, -0.46, 1.06)), 0.025), (m((0.07, -0.40, 1.16)), 0.008)])   # horn
    S.chain(ids[0], [((0, 0.46, 0.50), 0.1), ((0, 0.64, 0.44), 0.07), ((0, 0.82, 0.36), 0.05),
                     ((0, 1.0, 0.28), 0.035), ((0, 1.16, 0.22), 0.012)])
    for sg in (1, -1):
        m = (lambda p: p) if sg > 0 else mirror_x
        S.chain(ids[2], [(m((0.14, -0.14, 0.44)), 0.07), (m((0.18, -0.16, 0.24)), 0.05),
                         (m((0.18, -0.12, 0.06)), 0.04), (m((0.18, -0.22, 0.02)), 0.02)])  # front leg
        S.chain(ids[0], [(m((0.15, 0.28, 0.42)), 0.09), (m((0.18, 0.18, 0.24)), 0.06),
                         (m((0.18, 0.30, 0.08)), 0.04), (m((0.18, 0.20, 0.02)), 0.02)])    # hind leg
        # wing: arm spar folded along the flank
        w0 = S.add(m((0.14, -0.14, 0.72)), 0.05, ids[2])
        S.chain(w0, [(m((0.40, -0.10, 0.96)), 0.035), (m((0.70, 0.05, 1.05)), 0.025),
                     (m((0.95, 0.30, 0.95)), 0.012)])
    return None, None


def salamander(S):
    spine = [((0, 0.25, 0.14), 0.09), ((0, 0.08, 0.15), 0.11), ((0, -0.10, 0.15), 0.11),
             ((0, -0.26, 0.15), 0.09)]
    ids = []
    for p, r in spine:
        ids.append(S.add(p, r, ids[-1] if ids else None))
    S.chain(ids[-1], [((0, -0.38, 0.15), 0.08), ((0, -0.50, 0.14), 0.08), ((0, -0.60, 0.12), 0.05),
                      ((0, -0.64, 0.11), 0.02)])
    S.chain(ids[0], [((0, 0.42, 0.12), 0.07), ((0, 0.60, 0.10), 0.05), ((0, 0.80, 0.08), 0.035),
                     ((0, 1.00, 0.06), 0.02), ((0, 1.14, 0.05), 0.008)])
    for sg in (1, -1):
        m = (lambda p: p) if sg > 0 else mirror_x
        S.chain(ids[2], [(m((0.12, -0.13, 0.13)), 0.045), (m((0.26, -0.16, 0.15)), 0.035),
                         (m((0.30, -0.20, 0.02)), 0.03), (m((0.33, -0.28, 0.01)), 0.015)])
        S.chain(ids[0], [(m((0.12, 0.25, 0.13)), 0.05), (m((0.27, 0.26, 0.15)), 0.04),
                         (m((0.31, 0.30, 0.02)), 0.03), (m((0.35, 0.22, 0.01)), 0.015)])
    return None, None


bpy.ops.wm.read_factory_settings(use_empty=True)
S = Skel()
{'spider': spider, 'scorpion': scorpion, 'drake': drake, 'salamander': salamander}[KIND](S)
me = bpy.data.meshes.new(KIND)
me.from_pydata(S.v, S.e, [])
ob = bpy.data.objects.new(KIND, me)
bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
sk = ob.modifiers.new('skin', 'SKIN')
sk.use_smooth_shade = True
for i, r in enumerate(S.r):
    me.skin_vertices[0].data[i].radius = (r, r)
me.skin_vertices[0].data[0].use_root = True
sub = ob.modifiers.new('sub', 'SUBSURF')
sub.levels = int(opt.get('subdiv', 1))
bpy.ops.object.modifier_apply(modifier='skin')
bpy.ops.object.modifier_apply(modifier='sub')
# a flat base colour per kind; a UV'd texture would only prove the texture path
col = {'spider': (0.12, 0.10, 0.09), 'scorpion': (0.45, 0.30, 0.12), 'drake': (0.55, 0.16, 0.08),
       'salamander': (0.35, 0.10, 0.05)}[KIND]
mat = bpy.data.materials.new(KIND)
mat.use_nodes = True
mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*col, 1.0)
me.materials.append(mat)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')
out = os.path.abspath(os.path.expanduser(opt['out']))
os.makedirs(os.path.dirname(out), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', use_selection=True, export_yup=True)
print('STANDIN', KIND, len(me.vertices), 'verts', len(me.polygons), 'faces ->', out)
