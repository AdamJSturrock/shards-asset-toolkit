# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""
meshy_rig_weight_review.py: check how a Meshy auto-rig skins a held prop
(weapon) and the hands, before the Blender stage.

    blender --background --factory-startup --python scripts/blender/meshy_rig_weight_review.py -- \
        <manifest.json> [size]

Manifest: JSON array of {"id", "glb" (rigged GLB), "outdir", "weapon_side": "R"|"L"|null}.
Writes <outdir>/<id>__weights_{front,three_quarter}.png: every vertex painted by
the group of its dominant bone (hand = green, forearm = yellow, upper arm =
orange, spine/hips = red, neck/head = magenta, legs = grey, other = white;
the weapon side is saturated, the other side pale). And <outdir>/<id>.weights.json:
bone list, and for the vertices beyond the weapon-side wrist (hand + held prop),
split into "in the fist" and "outside the fist" (the prop), the share of
vertices whose dominant bone is in each group and the total weight each bone
carries. A prop weighted to the hand/forearm only shows as hand/forearm there;
smearing shows up as spine/head/neck shares.
"""
import bpy, sys, json, os, re
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
targets = json.load(open(argv[0]))
SIZE = int(argv[1]) if len(argv) > 1 else 768

GROUPS = [  # (name, regex on lower-cased bone name without side tokens)
    ("hand", r"hand|finger|thumb|index|middle|ring|pinky"),
    ("forearm", r"forearm|lowerarm|elbow"),
    ("upperarm", r"upperarm|arm|shoulder|clavicle"),
    ("head", r"head|neck|jaw|eye"),
    ("legs", r"leg|thigh|calf|foot|toe|knee|shin|upleg"),
    ("spine", r"spine|hip|pelvis|chest|root|torso"),
]
COL = {"hand": (0.1, 0.85, 0.1), "forearm": (0.95, 0.9, 0.1), "upperarm": (1.0, 0.5, 0.05),
       "head": (0.95, 0.1, 0.9), "legs": (0.5, 0.5, 0.5), "spine": (0.9, 0.1, 0.1), "other": (1, 1, 1)}


def side_of(name):
    n = name.lower()
    if re.search(r"(^|[^a-z])(l|left)([^a-z]|$)|left|_l$|\.l$", n):
        return "L"
    if re.search(r"(^|[^a-z])(r|right)([^a-z]|$)|right|_r$|\.r$", n):
        return "R"
    return None


def group_of(name):
    n = re.sub(r"left|right", "", name.lower())
    for g, rx in GROUPS:
        if re.search(rx, n):
            return g
    return "other"


for t in targets:
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=t["glb"])
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    # skinned meshes only (factory-startup scenes also carry a default cube)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and any(md.type == "ARMATURE" for md in o.modifiers)]
    bones = {b.name: (arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local) for b in arm.data.bones}
    wside = t.get("weapon_side") or "R"
    hand = next((n for n in bones if group_of(n) == "hand" and side_of(n) == wside and "hand" in n.lower()), None)
    forearm = next((n for n in bones if group_of(n) == "forearm" and side_of(n) == wside), None)
    report = {"id": t["id"], "bones": sorted(bones), "hand_bone": hand, "forearm_bone": forearm}
    allv = []
    for m in meshes:
        me = m.data
        names = {vg.index: vg.name for vg in m.vertex_groups}
        if not me.color_attributes.get("wcol"):
            me.color_attributes.new("wcol", "FLOAT_COLOR", "POINT")
        ca = me.color_attributes["wcol"]
        for v in me.vertices:
            ws = [(names[g.group], g.weight) for g in v.groups if g.weight > 0 and g.group in names]
            dom = max(ws, key=lambda x: x[1])[0] if ws else None
            g = group_of(dom) if dom else "other"
            c = Vector(COL[g])
            if dom and side_of(dom) and side_of(dom) != wside:
                c = c * 0.45 + Vector((0.55, 0.55, 0.55))
            ca.data[v.index].color = (c.x, c.y, c.z, 1)
            allv.append((m.matrix_world @ v.co, ws, dom))
    # region beyond the weapon-side wrist (hand head = wrist)
    if hand:
        wrist = bones[hand][0]
        elbow = bones[forearm][0] if forearm else None
        axis = (wrist - elbow).normalized() if elbow else Vector((1 if wside == "L" else -1, 0, 0))
        zs = [p.z for p, _, _ in allv]
        H = max(zs) - min(zs)
        sel = [(p, ws, dom) for p, ws, dom in allv if (p - wrist).dot(axis) > -0.01 * H]
        # bone tails from the glTF importer are guesses, so size the fist from
        # the body height (a clenched fist is about 1/10 of a humanoid's height)
        fist_r = 0.055 * H
        def summarise(vs):
            share, tot = {}, {}
            for p, ws, dom in vs:
                g = group_of(dom) if dom else "none"
                share[g] = share.get(g, 0) + 1
                for n, w in ws:
                    tot[n] = tot.get(n, 0) + w
            k = max(len(vs), 1)
            return {"verts": len(vs), "dominant_group_share": {g: round(c / k, 3) for g, c in sorted(share.items(), key=lambda x: -x[1])},
                    "bone_weight_mean": {n: round(w / k, 3) for n, w in sorted(tot.items(), key=lambda x: -x[1])[:10]}}
        # "outside the fist": perpendicular distance from the hand axis line beyond the fist radius
        def perp(p):
            d = p - wrist
            return (d - axis * d.dot(axis)).length
        report["beyond_wrist_in_fist"] = summarise([s for s in sel if perp(s[0]) <= fist_r])
        report["beyond_wrist_prop"] = summarise([s for s in sel if perp(s[0]) > fist_r])
        report["fist_radius_m"] = round(fist_r, 4)
    os.makedirs(t["outdir"], exist_ok=True)
    json.dump(report, open(os.path.join(t["outdir"], f"{t['id']}.weights.json"), "w"), indent=2)
    # render
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = sc.render.resolution_y = SIZE
    sc.render.film_transparent = True
    sh = sc.display.shading
    sh.light = "FLAT"; sh.color_type = "VERTEX"
    try: sc.view_settings.view_transform = "Standard"
    except Exception: pass
    pts = [p for p, _, _ in allv]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    ctr = (lo + hi) / 2
    cam = bpy.data.objects.new("c", bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"; cam.data.ortho_scale = max(hi - lo) * 1.1
    for name, d in [("front", Vector((0, -1, 0))), ("three_quarter", Vector((-1, -1, 0.9)).normalized())]:
        f = -d; r = f.cross(Vector((0, 0, 1))).normalized(); u = r.cross(f)
        cam.location = ctr - f * 10; cam.rotation_euler = Matrix((r, u, -f)).transposed().to_euler()
        cam.data.clip_end = 50
        sc.render.filepath = os.path.join(t["outdir"], f"{t['id']}__weights_{name}.png")
        bpy.ops.render.render(write_still=True)
    print(f"[weights] {t['id']}: hand={hand} prop={report.get('beyond_wrist_prop', {}).get('dominant_group_share')}")
