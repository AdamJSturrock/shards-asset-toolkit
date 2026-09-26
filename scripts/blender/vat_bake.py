# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""
vat_bake.py: bake a skinned GLB into a Vertex Animation Texture (VAT).

EXPERIMENTAL. The Shards of Stone game does not render units through VAT
today: units ship as skinned GLBs drawn with standard GPU skinning plus
instancing. This baker works, but treat its output as a research path, and
measure before you rely on it. Files named src/game/... below live in the
Shards of Stone game repo, not in this toolkit: they describe the runtime
contract the bake was written against.

Invoked by `shards-asset bake-vat` as:

    blender --background --factory-startup \
            --python scripts/blender/vat_bake.py -- \
            --input <character.glb> \
            --output-dir <tmp> \
            --unit-id <id> \
            [--animations <animations.glb>] \
            [--fps 30] [--max-frames 256]

Outputs in <tmp>:
    <unit-id>_mesh.glb       Static mesh + UV2 vertex-id lookup
    <unit-id>_vat_pos.png    16-bit RGBA, width=vertexCount, height=totalFrames
    <unit-id>_vat_norm.png   16-bit RGBA, same layout
    <unit-id>_vat.json       {version: 2, unitId, vertexCount, textureWidth,
                              textureHeight, fps, maxFrames, framesDownsampled,
                              clips:{name:{startFrame, frameCount, loop,
                                durationSec, sampleRateFps, sourceFrameCount,
                                sourceFrameStart, sourceFrameEnd, downsampled,
                                sourceAction, role?}},
                              clipWarnings:[], unmatchedActions:[],
                              axisHint: "blender_z_up: …"}

                              PLAYBACK CONTRACT (schema v2): a clip's baked row
                              count is not its authored length once the
                              --max-frames budget bites, so `frameCount / fps`
                              is the WRONG duration. Play a clip over
                              `durationSec` seconds, i.e. advance texture rows
                              at `sampleRateFps` (= frameCount / durationSec).

Axis convention:
    Blender is Z-up. Shards of Stone's Three.js scene is ALSO +Z-up
    (perspectiveCamera.up = (0,0,1); see World3DRenderer). We bake positions
    in object space using Blender's native axes (`v.co` reads) — i.e. the
    bake is +Z-up.
    The static mesh GLB that ships alongside is exported with
    `export_yup=True` (required by glTF spec compliance and other tools),
    so the bind-pose vertices in `<unitId>_mesh.glb` are Y-up. The runtime
    VAT shader (`src/game/rendering/shaders/vatVertex.glsl`) NEVER samples
    the geometry's `position` attribute — it uses `uv2.x` as a lookup
    index into the position texture, so the mesh's bind-pose frame is
    irrelevant. The shader writes `worldPos = instanceMatrix * vec4(
    samplePos, 1.0)` directly, and since both `samplePos` and
    `instanceMatrix` are in scene +Z-up, NO xzy swap is required.
    Standard reference VAT pipelines (OpenVAT, flanb/VAT-blender-addon)
    DO swap because their target engines are Y-up; we do not, because
    ours is Z-up. The shader-side convention is documented in the JSON
    metadata's `axisHint` field and at the top of vatVertex.glsl.

Clip name mapping:
    The Animation Action names embedded in source GLBs vary wildly (Meshy uses
    e.g. "CharacterArmature|CharacterArmature|Idle", or bare "Dead" / "Running"
    / "Double_Combo_Attack"). We map them to the canonical Shards of Stone clip
    names using EXACTLY the same alias table and normalization the runtime GLB
    path uses — see RUNTIME_STATE_ALIASES below.
    Missing clips are skipped gracefully — the JSON omits them and notes
    `availableActions` so downstream tooling can audit the gap — but a slot that
    is empty *while the source ships an action the runtime would have matched*
    is a bug, not a gap, and is reported as a loud WARNING in stdout and in the
    JSON's `clipWarnings` (printed by the bake).

References / prior art consulted:
    - OpenVAT (Houdini-style baker ported to Blender)
        https://github.com/JoshRBogart/unreal_tools (original Unreal version)
    - flanb/VAT-blender-addon  https://github.com/flanb/VAT-blender-addon
    - flement/VAT-blender-addon https://github.com/flement/VAT-blender-addon
    - Three.js VAT reference: https://threejs.org/examples/?q=vert#webgl_morphtargets_horse
    - mikelyndon r3f-webgl-vertex-animation-textures (runtime side)
        https://github.com/mikelyndon

We implement the bake from scratch using only `bpy` + `bmesh` so we don't have
to depend on an installed addon — the same approach used by
flanb/VAT-blender-addon's `vat_bake.py`. The mesh-export side uses Blender's
built-in glTF exporter.
"""

# ── Argument parsing (runs before bpy import for fast --help/--version) ────

import sys
import os
import json
import argparse
import struct
import zlib
import math
import re

# Anything after a bare `--` is forwarded to us by Blender's CLI.
def _own_args():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []

def parse_args():
    p = argparse.ArgumentParser(prog="blender_vat_bake")
    p.add_argument("--input", required=True, help="Path to character.glb")
    p.add_argument("--animations", default=None, help="Optional separate animations.glb")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--unit-id", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--target-verts", type=int, default=4000,
                   help="LOD-decimate skinned meshes down to ~this vertex count "
                        "before baking (default 4000). Pass 0 to disable.")
    return p.parse_args(_own_args())

ARGS = parse_args()

# ── Blender imports (lazy — fail fast outside Blender) ─────────────────────

try:
    import bpy           # type: ignore
    import bmesh         # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError:
    sys.stderr.write(
        "ERROR: This script must be run inside Blender:\n"
        "  blender --background --python scripts/blender/vat_bake.py -- ...\n"
    )
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════
#  ⚠ MIRROR TABLE — KEEP IN SYNC WITH THE RUNTIME ⚠
#
#  This is a VERBATIM transcription of `STATE_ALIASES` in
#      src/game/rendering/ModelManager.ts   (search: "const STATE_ALIASES")
#  which is what the *GLB* render path uses to bind source animation clips to
#  canonical states. Python and TypeScript can't share the table, so it is
#  duplicated here — and ModelManager.ts carries the reciprocal comment naming
#  THIS file. If you edit one, edit the other in the same commit.
#
#  History (why this comment is this loud): the baker used to carry its own
#  hand-rolled needles — death was ["death", "die", "killed"], which does NOT
#  match Meshy's actual action name "Dead" ("die" is not a substring of
#  "dead"). 169 of 269 shipped bakes had no death clip and nothing warned.
#  Same class of miss for walk: the neutral-critter GLBs ship a single action
#  named "Armature|Unreal Take|baselayer", which the old needles missed
#  entirely, so those units baked to a one-frame "rest" pose.
#
#  Matching also has to normalize the same way the runtime does — lowercase,
#  strip every non-alphanumeric — or aliases like "walkcycle" / "unrealtake"
#  can never match "Walk_Cycle" / "Unreal Take". See _normalize_clip_name().
# ══════════════════════════════════════════════════════════════════════════
RUNTIME_STATE_ALIASES = {
    "idle": ["idle", "confusedscratch"],
    # 'baselayer' / 'unrealtake' / 'unreal' catch Meshy's default animation
    # export name ("Armature|Unreal Take|baselayer") used when the model has a
    # single primary loop with no editorial label.
    "walk": ["walk", "walking", "walkcycle", "move", "moving",
             "baselayer", "unrealtake", "unreal"],
    "attack": ["attack", "attacking", "hit", "swing", "strike"],
    "death": ["death", "die", "dying", "dead"],
    # 'soell' covers the Meshy export typo in `mage_soell_cast_4`.
    "gather": ["gather", "harvest", "magespellcast", "magesoellcast",
               "spellcast", "soellcast", "mage"],
    # Jetpack units (the Ironjumper) mid-hop. NOT aliased to 'hop' — 'chop'
    # contains it, which would drag every woodcutting clip into this pool.
    "hop": ["jump", "leap"],
    # The runtime also defines special_1..3 aliases (['special1','ability1',
    # 'spell1','cast1'] etc). The VAT baker does NOT bake those — its spare
    # slots (extra_1..3, see EXTRA_SLOT_NAMES) are currently reserved for the
    # idle pool. Source actions named special_* therefore do not reach the VAT
    # path; they are reported in `unmatchedActions` in the JSON so the gap is
    # visible rather than silent.
    # Optional skeletal-path states (first shipped by the Blender-authored
    # goblin worker). The VAT baker has NO slot for them — like 'hop', a
    # source that ships one is reported as a CLIP-SLOT MISS by the guard in
    # main(), which is accurate: a VAT unit would not play it. Never a bare
    # 'sit' ('transition', 'situps').
    "build": ["build", "repair", "construct", "hammer"],
    "sleep": ["sleep", "asleep"],
    "sitting": ["sitting", "seated"],
}

# ── Baker-only fallback aliases (LOWER priority than every runtime alias) ──
#
# ⚠ These are NOT in ModelManager.ts's STATE_ALIASES — they are a strictly
# additive last-resort tier, appended AFTER the runtime aliases so a runtime
# match always wins. They exist because dropping them outright would lose a
# clip that the previous baker did bake:
#
#   attack ← 'shoot' / 'fire'
#       ratmen_jezzail's ONLY attack action is "Draw_and_Shoot_Left". The
#       runtime GLB path can't bind it either — that is a RUNTIME alias gap,
#       not a baker one. When STATE_ALIASES gains 'shoot'/'fire', delete this
#       entry rather than leaving two sources of truth.
#   gather ← 'mine' / 'chop' / 'work' / 'pullradish'
#       Retained from the pre-parity needle list ('pullradish' is Meshy's
#       worker-ish clip, present on every peon/miner/peasant GLB but matched by
#       nothing in either table — so a worker's gather slot is currently filled
#       by 'mage_soell_cast_*' via the runtime aliases, which is what the
#       runtime itself picks for the GLB path). Same rule: if the runtime table
#       gains them, drop them here.
#
# Anything added here MUST be reported in the bake summary the same way a miss
# is, so the divergence stays visible instead of becoming folklore.
BAKER_FALLBACK_ALIASES = {
    "attack": ["shoot", "fire"],
    "gather": ["mine", "chop", "work", "pullradish"],
}


def _aliases_for(state):
    """Runtime aliases first (highest priority), baker fallbacks appended."""
    return list(RUNTIME_STATE_ALIASES[state]) + list(BAKER_FALLBACK_ALIASES.get(state, []))


# Canonical clip order + loop flags. Loop flags mirror STATE_LOOP_CONFIG in
# ModelManager.ts (walk/gather repeat, attack/death one-shot, idle repeats when
# it has no variants). The needles are NOT re-declared here — they come
# straight from RUNTIME_STATE_ALIASES so the two can't drift within this file.
CANONICAL_CLIPS = [
    # (canonical name, alias list (runtime aliases + baker fallbacks), loop flag)
    ("idle",   _aliases_for("idle"),   True),
    ("walk",   _aliases_for("walk"),   True),
    ("attack", _aliases_for("attack"), False),
    ("gather", _aliases_for("gather"), True),
    ("death",  _aliases_for("death"),  False),
]


def _normalize_clip_name(name):
    """Lowercase + strip non-alphanumerics.

    Byte-for-byte equivalent of `normalizeClipName()` in
    src/game/rendering/ModelManager.ts:
        name.toLowerCase().replace(/[^a-z0-9]/g, '')
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def log(msg):
    sys.stdout.write(f"[blender_vat_bake] {msg}\n")
    sys.stdout.flush()


def reset_scene():
    """Wipe the default scene; --factory-startup already gave us a clean slate
    but the default Cube/Camera/Light still exist."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    """Import a GLB. Returns the list of top-level objects that were added."""
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    after = set(bpy.data.objects.keys())
    added = [bpy.data.objects[name] for name in (after - before)]
    return added


def find_skinned_mesh():
    """Find the first mesh object whose data is bound to an armature."""
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        # Prefer meshes with an Armature modifier (skinned).
        if any(m.type == "ARMATURE" for m in obj.modifiers):
            return obj
    # Fallback: any mesh.
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            return obj
    return None


def all_meshes():
    """All MESH objects currently in the scene."""
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def remove_loose_mesh_vertices(mesh):
    """Do not count orphan vertices as renderable surface during reduction."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()


def decimate_meshes_to_target(target_verts):
    """LOD-reduce every mesh in the scene down to ~target_verts using Blender's
    Decimate modifier in COLLAPSE mode.

    Why bake-time and not source-time:
      Meshy exports characters at ~30k verts. VAT texture width = vertex count,
      so per-unit total is ~100MB (16-bit RGBA × frames). Reducing to ~4k verts
      brings the per-unit footprint to ~5MB without touching the source GLB
      (which we keep at full res for Meshy preview, hero close-ups, etc.).

    Key properties of COLLAPSE decimation we rely on:
      - UV layers survive (Blender preserves UV mapping via vertex-pair merge).
      - Bone weights / vertex groups survive (weights from collapsed pairs are
        averaged). This is critical because the bake applies the armature
        modifier and reads per-frame deformed positions — if weights were lost
        we'd get a static mesh.
      - The Armature modifier must remain ABOVE the Decimate modifier in the
        stack? No — we ADD Decimate after import (so it goes to the bottom of
        the stack, after Armature), then APPLY it. Apply order: bottom to top,
        but we apply Decimate as a stand-alone op so it operates on the
        un-deformed rest-pose mesh data. The armature modifier is left intact
        and continues to deform the (now lower-poly) mesh at evaluation time.

    No-op if the mesh is already at/under target_verts (skip cheaply for rigid
    mech units that arrive with low vert counts).
    """
    if not target_verts or target_verts <= 0:
        log("decimation: disabled (--target-verts 0)")
        return

    # Ensure we're in OBJECT mode for modifier ops.
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    for obj in all_meshes():
        if obj.data is None:
            continue
        remove_loose_mesh_vertices(obj.data)
        starting_count = len(obj.data.vertices)
        if starting_count <= target_verts:
            log(f"decimation: skip '{obj.name}' (already {starting_count} verts ≤ target {target_verts})")
            continue

        # Stash and remove Armature modifiers before applying Decimate. Blender
        # won't let you apply a modifier that sits BELOW an enabled deformer
        # (Armature) in the stack — and we want Decimate to operate on the
        # rest-pose mesh data, not the deformed evaluation, so it's safest to
        # detach Armature, apply Decimate, and re-attach Armature. Vertex
        # groups (bone weights) live on the mesh data and survive the
        # round-trip; the Decimate modifier in COLLAPSE mode propagates them
        # to merged vertices automatically.
        stashed_armatures = []
        for m in list(obj.modifiers):
            if m.type == "ARMATURE":
                stashed_armatures.append({
                    "name": m.name,
                    "object": m.object,
                    "use_vertex_groups": m.use_vertex_groups,
                    "use_bone_envelopes": m.use_bone_envelopes,
                    "use_deform_preserve_volume": m.use_deform_preserve_volume,
                })
                obj.modifiers.remove(m)

        # Blender's Decimate ratio is a FACE ratio, not a vertex ratio. On
        # character meshes the actual vertex reduction lands at roughly
        # 2-3x the ratio (shared loops survive), so a single naive pass with
        # ratio=target/current overshoots. Iterate up to 4 passes, each time
        # recomputing the ratio against the current count, until we hit
        # target_verts (or fall below 10% of target — over-decimation guard).
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        passes = 0
        max_passes = 4
        floor_count = max(1, target_verts // 10)
        applied_ok = True
        while passes < max_passes:
            current = len(obj.data.vertices)
            if current <= target_verts:
                break
            ratio = target_verts / float(current)
            # Reduce the actual surface, without the old aggressive 0.35/0.6
            # multipliers. Orphan vertices are removed after each pass so
            # their count cannot trigger another destructive reduction.
            ratio = max(0.1, min(0.9, ratio))

            mod = obj.modifiers.new(name="LOD_Decimate", type="DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            mod.use_collapse_triangulate = True

            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as e:
                log(f"decimation: WARN apply failed on '{obj.name}' pass {passes}: {e}")
                applied_ok = False
                try:
                    obj.modifiers.remove(mod)
                except Exception:
                    pass
                break

            remove_loose_mesh_vertices(obj.data)
            new_count = len(obj.data.vertices)
            log(f"decimation: '{obj.name}' pass {passes}: {current} → {new_count} verts (face ratio {ratio:.4f})")
            passes += 1

            # Guard against over-decimation: if a single pass drops us under
            # the floor, stop here — silhouette is already at risk.
            if new_count <= floor_count:
                log(f"decimation: '{obj.name}' hit floor ({floor_count}); stopping early")
                break
            # If a pass made no progress (very rare — usually means the mesh
            # is already at minimum decimation), bail to avoid infinite loop.
            if new_count >= current:
                log(f"decimation: '{obj.name}' no progress this pass; stopping")
                break

        # Re-attach the Armature modifier(s) we stashed.
        for stash in stashed_armatures:
            new_mod = obj.modifiers.new(name=stash["name"], type="ARMATURE")
            new_mod.object = stash["object"]
            new_mod.use_vertex_groups = stash["use_vertex_groups"]
            new_mod.use_bone_envelopes = stash["use_bone_envelopes"]
            new_mod.use_deform_preserve_volume = stash["use_deform_preserve_volume"]

        if applied_ok:
            final_count = len(obj.data.vertices)
            log(f"decimation: '{obj.name}' final: {starting_count} → {final_count} verts ({passes} pass(es))")


def _rank_actions_for_state(all_actions, aliases):
    """Every action matching `aliases`, best candidate first.

    Ranking is (alias index, -frame count, name):
      1. Alias position — the runtime's alias lists are written most-specific
         first, so 'walk'/'walking' beat the catch-all 'baselayer'/'unreal'
         fallbacks. Without this, adding the Meshy default-export aliases would
         let a 300-frame "Armature|Unreal Take|baselayer" blob outrank the real
         "Walking" cycle on units that ship both.
      2. Frame count (longest wins) — the original heuristic, retained as the
         tie-break within one alias tier.
      3. Name — pure determinism so re-bakes are reproducible.
    """
    scored = []
    for act in all_actions:
        norm = _normalize_clip_name(act.name)
        for i, alias in enumerate(aliases):
            if alias in norm:
                scored.append((i, -_action_frame_count(act), act.name, act))
                break
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in scored]


def _runtime_slot_candidates(all_actions):
    """{state: [action names]} for every action the RUNTIME alias rules accept.

    Independent of CANONICAL_CLIPS on purpose — this is the yardstick the
    missing-slot guard measures the actual bake against."""
    out = {}
    for state in RUNTIME_STATE_ALIASES:
        ranked = _rank_actions_for_state(all_actions, _aliases_for(state))
        if ranked:
            out[state] = [a.name for a in ranked]
    return out


def collect_actions_for_object(armature_obj):
    """Return (matched, extra_idles, seen_names, slot_candidates):
      - matched: {canonical_name: action} for the five canonical clips,
      - extra_idles: idle-variant actions (backflip, jump…) for the spare
        extra_* slots — everything matching 'idle' except the base idle,
      - seen_names: all action names (diagnostics),
      - slot_candidates: {canonical_name: [action names]} — every action the
        RUNTIME alias rules would accept for that slot. Used by the missing-slot
        guard in main() so a silently-dropped clip becomes a loud WARNING."""
    all_actions = list(bpy.data.actions)
    seen_names = [a.name for a in all_actions]

    matched = {}
    for canonical, aliases, _loop in CANONICAL_CLIPS:
        ranked = _rank_actions_for_state(all_actions, aliases)
        if ranked:
            matched[canonical] = ranked[0]

    # Guard input — deliberately computed from RUNTIME_STATE_ALIASES and NOT
    # from CANONICAL_CLIPS. If the two ever disagree (someone edits the bake
    # table, or a future refactor breaks the matching logic), the guard in
    # main() still sees what the RUNTIME would have matched and shouts. A guard
    # that reuses the same table it is checking can only ever agree with itself
    # — which is precisely how the death-clip loss survived 269 bakes.
    slot_candidates = _runtime_slot_candidates(all_actions)

    # Idle pool: prefer an action named exactly 'idle' as the base (our merge
    # tool emits the primary idle under that exact name); every other
    # idle-matching action becomes a fun variant bound to an extra_* slot.
    idle_acts = _rank_actions_for_state(all_actions, RUNTIME_STATE_ALIASES["idle"])
    base_idle = next((a for a in idle_acts if _strip_dup_suffix(a.name) == "idle"), None)
    if base_idle is None:
        base_idle = matched.get("idle")
    if base_idle is not None:
        matched["idle"] = base_idle  # never let a variant claim the canonical slot
    # Variants = idle-matching actions that aren't the base AND aren't just a
    # Blender-duplicated base ('idle.001', from loading both character.glb and
    # animations.glb — both ship the base idle).
    extra_idles = [a for a in idle_acts
                   if a is not base_idle and _strip_dup_suffix(a.name) != "idle"]

    return matched, extra_idles, seen_names, slot_candidates


def _strip_dup_suffix(name):
    """Lowercase a name and drop Blender's '.001'/'.002' duplicate suffix."""
    n = name.lower()
    if "." in n:
        head, tail = n.rsplit(".", 1)
        if tail.isdigit():
            return head
    return n


def _action_frame_count(action):
    fr = action.frame_range
    return max(1, int(round(fr[1] - fr[0])) + 1)


def assign_action(armature_obj, action):
    """Make `action` the active action on the armature."""
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    animation = armature_obj.animation_data
    animation.use_nla = False
    animation.action = action
    # Blender 4.4+ actions carry explicit slots. Imported animation GLBs bind
    # their slot to Armature.001; assigning only .action on Armature otherwise
    # produces a static bake while every clip name looks correct.
    if hasattr(action, "slots") and len(action.slots):
        slot = next((s for s in action.slots if s.target_id_type == "OBJECT"), action.slots[0])
        animation.action_slot = slot
    animation.action_blend_type = "REPLACE"
    animation.action_influence = 1.0
    armature_obj.update_tag()



def evaluate_mesh_at_frame(mesh_obj, depsgraph, frame):
    """Set the timeline to `frame`, evaluate the depsgraph, and return a fresh
    Mesh with armature deformation applied (object space).

    Caller is responsible for `bpy.data.meshes.remove(...)` on the result."""
    bpy.context.scene.frame_set(frame)
    depsgraph.update()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    # to_mesh() returns a Mesh with all modifiers applied. We make a copy so it
    # outlives the depsgraph evaluation.
    eval_mesh = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=False, depsgraph=depsgraph)
    return eval_mesh


def write_png_rgba16(path, width, height, pixels):
    """Write a 16-bit-per-channel RGBA PNG without depending on PIL/numpy.

    `pixels` must be a flat sequence of length width*height*4, each value an
    int in [0, 65535] (big-endian when packed into the PNG row bytes).

    PNG 16-bit RGBA is broadly supported by browsers (Three.js loads it as a
    half-float-compatible 16-bit texture via the standard ImageLoader on
    Chrome/Safari/Firefox). We chose PNG over EXR because Three.js's default
    loader handles PNG out of the box, and 16 bits per channel (~5 decimal
    digits per axis after normalization) is sufficient resolution for
    character-scale meshes baked into a [-1, 1] normalized box.
    """
    if len(pixels) != width * height * 4:
        raise ValueError(f"pixel buffer size {len(pixels)} != {width*height*4}")

    # Build raw image data: each row prefixed by a filter byte (0 = None).
    bytes_per_row = width * 4 * 2  # 4 channels × 2 bytes
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type
        row_start = y * width * 4
        for x in range(width * 4):
            v = pixels[row_start + x] & 0xFFFF
            raw.append((v >> 8) & 0xFF)
            raw.append(v & 0xFF)

    def chunk(tag, data):
        length = struct.pack(">I", len(data))
        crc_input = tag + data
        crc = struct.pack(">I", zlib.crc32(crc_input) & 0xFFFFFFFF)
        return length + crc_input + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 6, 0, 0, 0)  # 16-bit RGBA, no interlace
    idat = zlib.compress(bytes(raw), 6)

    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def normalize_to_unit_box(positions):
    """Given a flat list of (x,y,z) tuples, return a (positions_normalized,
    bounds) where positions are in [-1, 1] and bounds = {min, max, center,
    scale}. The runtime undoes the normalization with the inverse transform
    stored in the JSON metadata."""
    if not positions:
        return [], {"min": [0,0,0], "max":[0,0,0], "center":[0,0,0], "scale": 1.0}
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    cx, cy, cz = ((mn[i] + mx[i]) * 0.5 for i in range(3))
    ex, ey, ez = ((mx[i] - mn[i]) * 0.5 for i in range(3))
    scale = max(ex, ey, ez)
    if scale <= 1e-9:
        scale = 1.0  # degenerate mesh (single vertex); avoid div-zero
    norm = []
    for x, y, z in positions:
        norm.append(((x - cx) / scale, (y - cy) / scale, (z - cz) / scale))
    return norm, {
        "min": list(mn), "max": list(mx),
        "center": [cx, cy, cz], "scale": scale,
    }


def float_to_u16(v_minus1_to_1):
    """Map [-1, 1] → [0, 65535]. Clamps out-of-range values (e.g. exaggerated
    death-pose extremities) to the boundary."""
    v = (v_minus1_to_1 + 1.0) * 0.5
    if v < 0.0: v = 0.0
    if v > 1.0: v = 1.0
    return int(round(v * 65535.0))


# ── Main bake ──────────────────────────────────────────────────────────────

def main():
    log(f"unit_id={ARGS.unit_id} input={ARGS.input}")
    reset_scene()

    # Set the scene fps so frame_set semantics line up with our sampling.
    bpy.context.scene.render.fps = ARGS.fps
    bpy.context.scene.render.fps_base = 1.0

    # Import character.
    import_glb(ARGS.input)
    # Optional separate animations file (Meshy convention).
    if ARGS.animations and os.path.exists(ARGS.animations):
        log(f"importing animations file: {ARGS.animations}")
        import_glb(ARGS.animations)

    # LOD-decimate skinned meshes BEFORE the bake loop. Decimating after the
    # armature is attached but before bake means per-frame deformation
    # evaluates against the reduced mesh, so the VAT texture width (=vertex
    # count) drops proportionally. Source GLBs ship at ~30k verts (Meshy
    # default) → ~100MB textures; target ~4k verts → ~5MB textures.
    decimate_meshes_to_target(ARGS.target_verts)

    mesh_obj = find_skinned_mesh()
    if mesh_obj is None:
        log("ERROR: no MESH object found after import.")
        sys.exit(1)

    # Facing fixes are authored as `__rotate_glb_wrapper__` root nodes
    # (the game repo's facing-fix tools). The importer turns each one into a
    # parent empty, and `v.co` below is read in the mesh's OWN object space — so
    # without this the wrapper never reached the bake, and every wrapper-fixed
    # static vehicle baked sideways while its GLB faced forward. Only those
    # wrappers are composed in; every other parent transform is treated exactly
    # as the bake always has.
    wrapper_rot = None
    ancestor = mesh_obj.parent
    while ancestor is not None:
        if ancestor.name.startswith("__rotate_glb_wrapper__"):
            rot = ancestor.matrix_basis.to_3x3().normalized()
            wrapper_rot = rot if wrapper_rot is None else rot @ wrapper_rot
        ancestor = ancestor.parent
    if wrapper_rot is not None:
        log(f"composing facing wrapper rotation into baked positions/normals: {[list(row) for row in wrapper_rot]}")

    armature_obj = None
    for m in mesh_obj.modifiers:
        if m.type == "ARMATURE" and m.object is not None:
            armature_obj = m.object
            break
    if armature_obj is None:
        # Some Meshy exports parent the mesh to the armature instead of using a modifier.
        if mesh_obj.parent and mesh_obj.parent.type == "ARMATURE":
            armature_obj = mesh_obj.parent

    log(f"mesh={mesh_obj.name} verts={len(mesh_obj.data.vertices)} armature={armature_obj.name if armature_obj else '<none>'}")

    if armature_obj is None:
        log("WARN: no armature found — output will be a static mesh with one frame per clip.")

    # Discover matching actions.
    actions = {}
    extra_idles = []
    seen_action_names = []
    slot_candidates = {}
    if armature_obj is not None:
        actions, extra_idles, seen_action_names, slot_candidates = \
            collect_actions_for_object(armature_obj)
    else:
        # No armature: bpy.data.actions may still hold actions (e.g. the import
        # attached them to a different object). Record them so the guard below
        # can still report the loss instead of silently emitting a rest pose.
        seen_action_names = [a.name for a in bpy.data.actions]
        slot_candidates = _runtime_slot_candidates(list(bpy.data.actions))
    log(f"actions seen in source: {seen_action_names}")
    log(f"actions matched to canonical clips: {sorted(actions.keys())}")

    # Build a stable vertex index → object-space rest position table from the
    # original (un-evaluated) mesh data. We'll bake one row per frame; the
    # column index = vertex index.
    n_verts = len(mesh_obj.data.vertices)
    if n_verts == 0:
        log("ERROR: mesh has zero vertices.")
        sys.exit(1)

    # Determine clip layout. If no actions matched, emit a single "rest" pose
    # so downstream tooling still has *something* to work with.
    # Generic spare slots (must match VAT_CLIP_NAMES in VATTextureCache.ts).
    EXTRA_SLOT_NAMES = ["extra_1", "extra_2", "extra_3"]
    clip_roles = {}  # slot_name -> role string ('idle' for idle-pool variants)

    clip_layout = []  # list of (clip_name, action_or_None, frame_count, loop)
    if not actions:
        clip_layout.append(("rest", None, 1, True))
    else:
        for canonical, _needles, default_loop in CANONICAL_CLIPS:
            if canonical in actions:
                act = actions[canonical]
                fc = _action_frame_count(act)
                # Some authoring tools emit a single-frame clip for "T-pose";
                # keep at least 1 frame.
                fc = max(1, fc)
                clip_layout.append((canonical, act, fc, default_loop))
        # Idle-pool variants → the generic extra_* slots, tagged role 'idle'
        # so the runtime folds them into the idle cycler. Looping clips.
        for i, act in enumerate(extra_idles[:len(EXTRA_SLOT_NAMES)]):
            slot = EXTRA_SLOT_NAMES[i]
            clip_layout.append((slot, act, max(1, _action_frame_count(act)), True))
            clip_roles[slot] = "idle"
        if extra_idles:
            log(f"idle pool: base 'idle' + {len(clip_layout) - len(actions)} variant(s) "
                f"→ {[n for n in clip_roles]}")

    # ── Missing-slot guard ──────────────────────────────────────────────────
    #
    # The root cause of the "169 bakes shipped with no death clip" incident was
    # not the bad alias — it was that nothing complained. If the source ships an
    # action that the RUNTIME alias rules would bind to a canonical state, and
    # that state did not make it into the layout, say so LOUDLY. bake_vat.js
    # greps stdout for the 'CLIP-SLOT MISS' marker and re-prints these in its
    # per-unit line and end-of-run summary; they're also written to the JSON as
    # `clipWarnings` so a later audit pass can find them without the logs.
    laid_out = {name for name, _a, _fc, _l in clip_layout}
    clip_warnings = []
    # Iterate the RUNTIME states, not CANONICAL_CLIPS — a state dropped from the
    # bake table entirely must still be checked.
    for canonical in RUNTIME_STATE_ALIASES:
        if canonical in laid_out:
            continue
        candidates = slot_candidates.get(canonical) or []
        if not candidates:
            continue  # genuinely absent in the source — a gap, not a loss
        clip_warnings.append({
            "slot": canonical,
            "reason": "no clip baked into this slot although the source ships "
                      "matching action(s) under the runtime alias rules",
            "candidates": candidates,
        })
        log(f"WARNING: CLIP-SLOT MISS '{canonical}' — source has "
            f"{candidates} but nothing was baked into the '{canonical}' slot. "
            f"Check RUNTIME_STATE_ALIASES parity with ModelManager.ts.")

    # Slots that only got filled thanks to BAKER_FALLBACK_ALIASES — i.e. the
    # runtime's own table would NOT have bound this action. Reported so the
    # divergence is visible in every bake log until the runtime table catches
    # up (see the BAKER_FALLBACK_ALIASES comment).
    fallback_alias_slots = []
    for name, act, _fc, _l in clip_layout:
        if act is None or name not in RUNTIME_STATE_ALIASES:
            continue
        norm_name = _normalize_clip_name(act.name)
        if not any(a in norm_name for a in RUNTIME_STATE_ALIASES[name]):
            fallback_alias_slots.append({"slot": name, "sourceAction": act.name})
            log(f"NOTE: slot '{name}' filled by baker-fallback alias only "
                f"(action '{act.name}' matches no runtime alias in "
                f"ModelManager.ts STATE_ALIASES).")

    # Actions that landed in no slot at all (informational — e.g. Meshy's
    # 'Pull_Radish', 'Zombie_Scream', 'special_1'). Not a warning: the runtime
    # doesn't bind them either. Emitted so a human auditing a unit can see what
    # the source had to offer.
    used_action_names = {a.name for _n, a, _fc, _l in clip_layout if a is not None}
    matched_any = set()
    for names in slot_candidates.values():
        matched_any.update(names)
    unmatched_actions = [n for n in seen_action_names
                         if n not in used_action_names and n not in matched_any]
    if unmatched_actions:
        log(f"unmatched actions (bound to no canonical slot): {unmatched_actions}")

    total_frames = sum(fc for _n, _a, fc, _l in clip_layout)
    frames_downsampled = False
    if total_frames > ARGS.max_frames:
        # Fit the frame budget by sampling each clip more SPARSELY — the bake
        # loop below always spreads its `fc` samples evenly across the action's
        # full [start, end] range, so coverage stays complete: the clip is
        # decimated in time, never truncated. (The old comment here claimed
        # truncation; it was wrong, and that wording is what made the too-fast
        # playback look like expected behaviour.)
        #
        # The consequence a reader must not miss: a clip's baked row count is
        # then NO LONGER its authored length, so `frameCount / fps` is the wrong
        # playback duration. Every clip therefore carries `durationSec` (the
        # authored length) and `sampleRateFps` (rows per second) in the JSON —
        # the runtime must advance rows at `sampleRateFps`, not at `fps`.
        scale = ARGS.max_frames / float(total_frames)
        new_layout = []
        for name, act, fc, loop in clip_layout:
            new_fc = max(1, int(math.floor(fc * scale)))
            new_layout.append((name, act, new_fc, loop))
        clip_layout = new_layout
        total_frames = sum(fc for _n, _a, fc, _l in clip_layout)
        frames_downsampled = True
        log(f"WARN: total frames > max ({ARGS.max_frames}); downsampled "
            f"(clips keep their authored durationSec; rows/sec = sampleRateFps).")

    log(f"clip_layout: {[(n, fc) for n, _a, fc, _l in clip_layout]}  total_frames={total_frames}")

    # ── Bake positions + normals ────────────────────────────────────────────

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # First pass: gather raw positions for normalization bounds.
    raw_positions = []  # flat list: total_frames rows × n_verts cols of (x,y,z)
    raw_normals   = []  # same layout

    cursor_frame = 0
    clips_out = {}
    for name, act, fc, loop in clip_layout:
        if armature_obj is not None and act is not None:
            assign_action(armature_obj, act)
            start_frame = int(round(act.frame_range[0]))
            end_frame   = int(round(act.frame_range[1]))
        else:
            start_frame = 1
            end_frame = 1

        # Sample `fc` evenly across [start_frame, end_frame].
        for i in range(fc):
            if fc == 1:
                src_frame = start_frame
            else:
                t = i / float(fc - 1) if fc > 1 else 0.0
                src_frame = int(round(start_frame + t * (end_frame - start_frame)))

            ev_mesh = evaluate_mesh_at_frame(mesh_obj, depsgraph, src_frame)
            # Blender 4.1 removed Mesh.calc_normals_split (normals are now
            # auto-computed on access). Call it only if present (Blender < 4.1).
            if hasattr(ev_mesh, "calc_normals_split"):
                ev_mesh.calc_normals_split()

            # Per-vertex (loop normals averaged back to per-vertex isn't
            # strictly correct lighting-wise, but VAT typically reconstructs
            # smooth normals from the position deltas at runtime anyway).
            # We use the per-vertex `.normal` straight off the eval mesh.
            for v in ev_mesh.vertices:
                # wrapper_rot: the facing wrapper(s) composed above, or None.
                co = v.co if wrapper_rot is None else wrapper_rot @ v.co
                no = v.normal if wrapper_rot is None else (wrapper_rot @ v.normal).normalized()
                raw_positions.append((co.x, co.y, co.z))
                raw_normals.append((no.x, no.y, no.z))

            bpy.data.meshes.remove(ev_mesh)

        # ── Authored-tempo metadata (schema v2) ─────────────────────────────
        #
        # `frameCount` is how many ROWS this clip occupies in the texture, which
        # after the frame-budget downsample is NOT its authored length. Playing
        # rows at `fps` therefore runs the clip fast (a 4s death played back in
        # ~2.9s — the bug this field exists to fix).
        #
        #   durationSec   authored wall-clock length of the source action.
        #   sampleRateFps rows/second the runtime must advance to hit that
        #                 length exactly = frameCount / durationSec.
        #
        # Both are written for EVERY clip, downsampled or not, so the runtime
        # never needs a branch. durationSec is always > 0 (single-frame poses
        # report one frame's worth of time).
        authored_span = max(1, end_frame - start_frame)
        authored_frame_count = max(1, end_frame - start_frame + 1)
        duration_sec = authored_span / float(ARGS.fps)
        clip_entry = {
            "startFrame": cursor_frame,
            "frameCount": fc,
            "loop": bool(loop),
            "durationSec": round(duration_sec, 6),
            "sampleRateFps": round(fc / duration_sec, 6),
            "sourceFrameCount": authored_frame_count,
            "sourceFrameStart": start_frame,
            "sourceFrameEnd": end_frame,
            "downsampled": bool(fc < authored_frame_count),
            "sourceAction": act.name if act is not None else None,
        }
        if name in clip_roles:
            clip_entry["role"] = clip_roles[name]
        clips_out[name] = clip_entry
        cursor_frame += fc
        log(f"  baked clip '{name}': {fc} frames "
            f"(authored {authored_frame_count} frames / {duration_sec:.3f}s "
            f"→ {clip_entry['sampleRateFps']:.3f} rows/sec)")

    assert len(raw_positions) == total_frames * n_verts, \
        f"position count mismatch: got {len(raw_positions)}, expected {total_frames * n_verts}"

    # Normalize positions to [-1, 1] over the full animation envelope so the
    # 16-bit precision is used efficiently.
    norm_positions, pos_bounds = normalize_to_unit_box(raw_positions)
    # Normals are already approximately unit vectors; clamp into [-1, 1] just
    # to be safe.
    norm_normals = [(max(-1.0, min(1.0, n[0])),
                     max(-1.0, min(1.0, n[1])),
                     max(-1.0, min(1.0, n[2]))) for n in raw_normals]

    # Pack into RGBA16 buffers (alpha = 1.0 always → 65535).
    tex_w = n_verts
    tex_h = total_frames
    pos_pixels = [0] * (tex_w * tex_h * 4)
    norm_pixels = [0] * (tex_w * tex_h * 4)
    for idx, (p, no) in enumerate(zip(norm_positions, norm_normals)):
        # Texture layout: row = frame, col = vertex index. PNGs are written
        # top-left origin; we keep frame 0 on row 0 for consistency.
        # idx already iterates frames-major (clip layout above), vertex-minor.
        # Within each frame, vertex 0 is at col 0.
        # Frame = idx // n_verts, Vertex = idx % n_verts.
        # We just need a flat dest index that matches: dest = idx * 4.
        d = idx * 4
        pos_pixels[d + 0] = float_to_u16(p[0])
        pos_pixels[d + 1] = float_to_u16(p[1])
        pos_pixels[d + 2] = float_to_u16(p[2])
        pos_pixels[d + 3] = 65535  # alpha = 1
        norm_pixels[d + 0] = float_to_u16(no[0])
        norm_pixels[d + 1] = float_to_u16(no[1])
        norm_pixels[d + 2] = float_to_u16(no[2])
        norm_pixels[d + 3] = 65535

    out_dir = ARGS.output_dir
    os.makedirs(out_dir, exist_ok=True)
    pos_path  = os.path.join(out_dir, f"{ARGS.unit_id}_vat_pos.png")
    norm_path = os.path.join(out_dir, f"{ARGS.unit_id}_vat_norm.png")
    log(f"writing {pos_path} ({tex_w}x{tex_h} 16-bit RGBA)")
    write_png_rgba16(pos_path, tex_w, tex_h, pos_pixels)
    log(f"writing {norm_path} ({tex_w}x{tex_h} 16-bit RGBA)")
    write_png_rgba16(norm_path, tex_w, tex_h, norm_pixels)

    # ── Export static mesh with UV2 vertex-id lookup ────────────────────────
    #
    # We add a second UV layer (UV2) whose .x carries (vertexIndex / vertexCount)
    # and .y is 0.0. The runtime vertex shader then samples the VAT at
    # (uv2.x, frameRow / textureHeight). This is the standard trick used by
    # OpenVAT, flanb/VAT-blender-addon, etc.
    #
    # We export from a copy of the mesh with all modifiers applied (so the
    # exported geometry is the rest-pose shape that matches frame 0 of every
    # baked clip), no armature (we don't need bones at runtime), and only the
    # UV2 channel of interest.

    log("preparing static-mesh export…")

    # Make a clean duplicate of the original mesh in rest pose. We use a fresh
    # bmesh and write the per-loop UV2 so it survives glTF export.
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="OBJECT")

    # Add a UV2 layer named "vat_id" that we then populate.
    mesh_data = mesh_obj.data
    if "vat_id" not in [uv.name for uv in mesh_data.uv_layers]:
        mesh_data.uv_layers.new(name="vat_id")
    vat_uv_layer = mesh_data.uv_layers["vat_id"]
    # Populate: each loop's UV2.x = its vertex index / n_verts; UV2.y = 0.5
    # (sampled at the row center is unnecessary — the runtime computes the row
    # from the clip frame, not from UV2.y. We park it at 0 for clarity).
    for poly in mesh_data.polygons:
        for loop_idx in poly.loop_indices:
            v_idx = mesh_data.loops[loop_idx].vertex_index
            u = (v_idx + 0.5) / float(n_verts)  # +0.5 → sample column center
            vat_uv_layer.data[loop_idx].uv = (u, 0.0)

    # Remove armature modifier so the exported mesh is genuinely static.
    for mod in list(mesh_obj.modifiers):
        if mod.type == "ARMATURE":
            mesh_obj.modifiers.remove(mod)

    # Detach from the armature parent (if any) so the glTF exporter doesn't
    # follow up into the skeleton.
    if mesh_obj.parent is not None and mesh_obj.parent.type == "ARMATURE":
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    # Hide the armature so the exporter doesn't include it.
    if armature_obj is not None:
        armature_obj.hide_set(True)
        armature_obj.hide_select = True

    # Select only the mesh.
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    mesh_glb_path = os.path.join(out_dir, f"{ARGS.unit_id}_mesh.glb")
    log(f"exporting {mesh_glb_path}")
    bpy.ops.export_scene.gltf(
        filepath=mesh_glb_path,
        export_format="GLB",
        use_selection=True,
        export_apply=False,         # we kept the rest pose as-is
        export_animations=False,
        export_skins=False,
        export_morph=False,
        export_yup=True,            # match Three.js coordinate convention
        export_texcoords=True,      # keep both UV layers (UV0 + UV2)
    )

    # ── Emit metadata JSON ──────────────────────────────────────────────────

    metadata = {
        # v2 (2026-08): per-clip `durationSec` + `sampleRateFps` added, and the
        #     clip→state matching was aligned with ModelManager.ts's
        #     STATE_ALIASES (v1 bakes silently dropped death/walk clips).
        #     v1 readers see v2 files as compatible — nothing was removed — but
        #     a v1 file has no durationSec, so a v2 runtime must fall back to
        #     `frameCount / fps` (i.e. the old, too-fast behaviour) for those.
        "version": 2,
        "unitId": ARGS.unit_id,
        "vertexCount": n_verts,
        "textureWidth": tex_w,
        "textureHeight": tex_h,
        # Sampling fps used at bake time. NOT a playback rate: use each clip's
        # `sampleRateFps` for playback (they differ whenever `downsampled`).
        "fps": ARGS.fps,
        "maxFrames": ARGS.max_frames,
        "framesDownsampled": bool(frames_downsampled),
        "clips": clips_out,
        # Loud, machine-readable record of clips the source had but the bake
        # did not produce. Empty list = clean bake. See the missing-slot guard.
        "clipWarnings": clip_warnings,
        "fallbackAliasSlots": fallback_alias_slots,
        "unmatchedActions": unmatched_actions,
        "positionBounds": pos_bounds,
        "encoding": {
            "format": "PNG_RGBA16",
            "channels": "RGB = position.xyz (normalized), A = 1.0",
            "rangeMapping": "(value / 65535) * 2 - 1, then * positionBounds.scale + positionBounds.center",
        },
        "axisHint": "blender_z_up: positions and normals are baked in Blender's native +Z-up object space (v.co reads). Shards of Stone's three.js scene is also +Z-up, so the runtime shader (src/game/rendering/shaders/vatVertex.glsl) does NOT apply an xzy swap. The static mesh GLB ships Y-up (export_yup=True) but the shader does not sample the geometry's `position` attribute, only uv2.x, so the mesh frame is irrelevant. If the receiving scene ever moves to Y-up, uncomment the `samplePos = samplePos.xzy;` line in vatVertex.glsl.",
        "availableActions": seen_action_names,
        "source": os.path.basename(ARGS.input),
        "sourceAnimations": os.path.basename(ARGS.animations) if ARGS.animations else None,
        "targetVerts": ARGS.target_verts,
        "bakedBy": "shards-asset-toolkit scripts/blender/vat_bake.py v2",
    }
    json_path = os.path.join(out_dir, f"{ARGS.unit_id}_vat.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"wrote {json_path}")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stderr.write(f"FATAL: {e}\n")
        sys.exit(1)
