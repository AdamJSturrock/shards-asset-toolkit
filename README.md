# Shards of Stone: 3D Asset Pipeline & Toolkit (`@shardsofstone/asset-toolkit`)

The official open-source generative 3D asset pipeline for [Shards of Stone](https://www.shardsofstone.com) map makers, modders, and custom game creators.

This toolkit automates the entire journey from a text prompt or 2D concept into a fully animated, browser-optimised 3D unit. It produces production-ready `.glb` models and Vertex Animation Texture (VAT) bundles that run at 60 fps in web browsers and Three.js scenes with zero manual 3D modeling required.

---

## What this toolchain does

Building custom maps with custom units usually requires expensive freelance artists or hundreds of hours in 3D modeling packages. 

This repository connects an automated chain:
1. **Concept generation & cleanup:** Generates isometric 3/4 orthographic unit concepts via Google Gemini, isolates silhouettes, and strips atmospheric effects (smoke, floating embers, ground pedestals) that break 3D generation.
2. **3D mesh generation:** Converts clean 2D concepts into 30,000-polygon textured GLBs via the Meshy API in under two minutes.
3. **Automated rigging & skinning:** Runs headless Blender scripts to fit anatomical skeletons, isolates vertex weights by limb to prevent mesh stretching, and generates procedural gait cycles (for crabs, spiders, scorpions, quadrupeds, dragons) or retargets motion capture.
4. **Humanoid mocap bridge:** Retargets free Mixamo FBX motion clips onto custom humanoid proportions (goblins, dwarves, ogres) while preserving character stance and isolating weapons.
5. **Browser RTS optimization:** Resizes textures to 1024px, transcodes them to GPU-native KTX2 Basis Universal formats, compresses geometry with `meshopt`, and bakes animations into 16-bit Vertex Animation Textures (VAT) for zero-CPU instanced crowd rendering in Three.js.

---

## Toolchain requirements

Before running the pipeline, ensure the following software is installed on your machine:

| Requirement | Supported versions | Notes |
|---|---|---|
| **Node.js** | `>= 20.6.0` | Powers the pipeline CLI and glTF transform tools |
| **Python** | `>= 3.10` | Used for auxiliary geometry processing and inspection |
| **Blender** | `4.2+` or `5.x` | Required for headless rigging, skinning, and VAT baking |
| **glTF-Transform CLI** | `^4.0.0` | Required for texture transcoding and meshopt compression |

### Installing dependencies

```bash
# Clone the repository
git clone https://github.com/AdamJSturrock/shards-asset-toolkit.git
cd shards-asset-toolkit

# Install Node dependencies
npm install

# Install glTF-Transform CLI globally (or use the bundled local binary)
npm install -g @gltf-transform/cli

# Ensure Blender is accessible from your terminal
# On macOS (Homebrew or Applications):
export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"
# Or on Ubuntu / Debian:
# sudo apt install blender
```

---

## Required API keys & services

The pipeline uses cloud APIs for image and mesh generation. You will need:

1. **Google Gemini API Key (`GEMINI_API_KEY`):**
   - Used for concept generation, candidate variations, and background cleanup.
   - Get a key from [Google AI Studio](https://aistudio.google.com/).
2. **Meshy API Key (`MESHY_API_KEY`):**
   - Used for converting 2D concept images into textured 3D meshes.
   - Get a key and credits from [Meshy.ai](https://www.meshy.ai/).
3. **Anthropic API Key (`ANTHROPIC_API_KEY`, optional):**
   - Used if running in agentic mode where Claude Opus 3.5 / 5.5 or Astra coordinates complex landmark fitting, diagnoses mesh topologies, and writes custom rigging scripts.
4. **Mixamo Account (free):**
   - [Mixamo by Adobe](https://www.mixamo.com) provides thousands of free bipedal animation clips (walk, run, attack, death, spellcast) in FBX format.

### Environment setup

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
MESHY_API_KEY=your_meshy_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender
```

---

## Cost breakdown & estimates per unit

How much does it actually cost to produce an animated game-ready 3D unit?

Because the pipeline automates each step and logs resource consumption, the costs are transparent and predictable:

| Stage | Service / Tool | Usage per unit | Estimated cost (USD) |
|---|---|---|---|
| **Concept art** | Gemini 3 Pro / Flash | 3-4 candidate generations + 1 isolation | $0.05 to $0.15 |
| **3D mesh generation** | Meshy API | 30 credits (standard 30k poly mesh) | $0.30 to $0.45 |
| **Agentic rigging & analysis** | Claude Opus / Astra | ~20,000 to 30,000 tokens (scripting & review) | $0.20 to $0.40 |
| **Animation authoring** | Blender / Mixamo | Local headless compute (~1-2 minutes) | $0.00 (Free) |
| **Web optimization & VAT** | gltf-transform + Blender | Local headless compute (~45 seconds) | $0.00 (Free) |
| **Total per finished unit** | | **All stages combined** | **~$0.55 to $1.00** |

### Comparison to traditional game development

- **Freelance contractor:** $300 to $1,500 per unit, 2 to 3 weeks delivery time.
- **This pipeline:** Under $1.00 per unit, 10 to 15 minutes end to end.
- **Budget for a full custom faction (12 units + 8 buildings):** Roughly $15 to $20 in API credits.

---

## How to use: step-by-step guide

You can run individual stages or execute the full end-to-end pipeline with a configuration file.

### 1. Generating concept art

Prompt Gemini for an isometric three-quarter orthographic concept with transparent or white background:

```bash
node bin/shards-asset.mjs concept \
  --prompt "A giant armored reef crab unit with coral on its carapace, heavy snapping claws, 8 walking legs, fantasy RTS style" \
  --output ./output/concepts/reef_crab.png \
  --strip-effects
```

**Key rule:** The `--strip-effects` flag removes floating sparks, smoke, dripping liquids, web strands, and ground pedestals. 3D generators turn atmospheric effects and ground terrain into solid geometry lumps that ruin the character mesh.

### 2. Generating the 3D mesh

Send the cleaned concept to Meshy to generate a 30,000-polygon textured GLB:

```bash
node bin/shards-asset.mjs mesh \
  --input ./output/concepts/reef_crab.png \
  --output ./output/meshes/reef_crab.glb \
  --polycount 30000 \
  --texture-size 2048
```

The script polls the Meshy task status until completion and downloads the resulting GLB and preview thumbnail.

### 3. Rigging & animating

#### Option A: Non-humanoids (Crabs, Spiders, Scorpions, Quadrupeds, Drakes)

Non-humanoids use our procedural rigging engine in headless Blender:

```bash
node bin/shards-asset.mjs rig-critter \
  --input ./output/meshes/reef_crab.glb \
  --template crustacean \
  --output ./output/rigged/reef_crab.glb
```

Available templates:
- `crustacean`: Carapace, 8 walking legs, 2 claws with snapping pincers, lateral scuttling gait.
- `arachnid`: Cephalothorax, abdomen, 8 legs (4 segments each), fangs, alternating tetrapod gait.
- `scorpion`: 8 walking legs, 2 pedipalps with pincers, 5-joint segmented stinging tail with strike attack.
- `sprawl`: Low-slung quadruped with lateral spine undulation and tail sway (salamanders, lizards).
- `drake`: Flying wing armature with hovering flight cycle, banked turn loop, and breath attack.

**Topological component isolation:** The script executes `fit_arthropod.py` to identify landmark seeds and trace geodesic edge distances. Vertices are tagged into isolated component masks so bone weights cannot bleed into neighbouring legs.

#### Option B: Humanoids (Goblins, Dwarves, Humans, Ogres)

For humanoids, download desired FBX animation clips from Mixamo (e.g. `idle.fbx`, `walk.fbx`, `attack.fbx`, `death.fbx`), then retarget them:

```bash
node bin/shards-asset.mjs retarget-mixamo \
  --target ./output/meshes/goblin_peon.glb \
  --clip idle=./mocap/goblin_peon/idle.fbx \
  --clip walk=./mocap/goblin_peon/walk.fbx \
  --clip attack=./mocap/goblin_peon/slash.fbx \
  --clip death=./mocap/goblin_peon/death.fbx \
  --leg-align 0.5 \
  --loop idle \
  --loop walk \
  --out ./output/rigged/goblin_peon.glb
```

- `--leg-align 0.5`: Preserves the character's unique anatomy (e.g. a goblin's bowed knees or an ogre's wide stance) instead of forcing rigid Mixamo posture.
- `--loop <name>`: Automatically cancels horizontal root motion drift so walk and run cycles stay centered at the origin.

### 4. Browser optimization pass

A raw 25 MB GLB will crash browser tabs when multiple armies clash. Run the optimization pass to produce a compact, production-ready asset:

```bash
node bin/shards-asset.mjs optimize \
  --input ./output/rigged/reef_crab.glb \
  --texture-max 1024 \
  --codec mixed \
  --output ./output/optimized/reef_crab.glb
```

This applies:
- Texture resizing to 1024px.
- KTX2 Basis Universal compression: UASTC for normal and metallic-roughness maps; ETC1S for diffuse colour maps.
- Geometry and keyframe track compression via `meshopt`.

### 5. Baking Vertex Animation Textures (VAT) for crowd rendering

For units that appear in large armies, skeletal skinning on the CPU is a major performance bottleneck. Baking Vertex Animation Textures allows Three.js to animate hundreds of units on the GPU in a single instanced draw call:

```bash
node bin/shards-asset.mjs bake-vat \
  --input ./output/rigged/reef_crab.glb \
  --output-dir ./output/baked/ \
  --target-verts 4000
```

Outputs generated:
- `<unit>_mesh.glb`: Static, unrigged decimated mesh (~4,000 vertices) with UV2 vertex lookups.
- `<unit>_vat_pos.ktx2`: 16-bit RGBA texture containing per-frame vertex coordinate deltas.
- `<unit>_vat_norm.ktx2`: Normal texture preserving dynamic lighting during animation.
- `<unit>_vat.json`: Sidecar metadata defining clip frame ranges, authored durations, and playback speeds.

---

## Full pipeline execution via config file

You can define a complete unit specification in JSON and execute all stages in one command:

```json
{
  "unitId": "reef_crab",
  "type": "critter",
  "template": "crustacean",
  "prompt": "A giant armored reef crab unit with coral on its carapace, heavy snapping claws, 8 walking legs, fantasy RTS style",
  "targetVerts": 4000,
  "textureMax": 1024,
  "vat": true
}
```

```bash
node bin/shards-asset.mjs pipeline --config examples/units/reef_crab.json
```

---

## Deep dive: the browser optimization pipeline

Understanding why these optimization steps exist will save you days of debugging web crashes.

### The mobile & browser reality

When running an RTS in a web browser using WebGL2 and Three.js:
1. **GPU Texture memory (VRAM):** Browsers must decode ordinary PNG/JPEG files into uncompressed 32-bit RGBA bitmaps in GPU memory. A single 2048px texture consumes 16 MB of VRAM. With 10 units having diffuse, normal, and roughness maps, your game consumes nearly 500 MB of VRAM on art alone. Mobile Safari and Chrome will kill the tab ("Aw Snap! Error 5").
2. **CPU Skeletal Matrix computation:** Animating 150 skeletal models requires JavaScript to update thousands of bone matrices per frame and upload them to the GPU. This overwhelms the browser's main thread and drops frame rates to single digits.
3. **Shader sampler limits:** WebGL2 guarantees only 16 texture units per shader. In Shards of Stone, the terrain shader already uses 16/16 samplers. Unit shaders must remain compact.

### Why KTX2 Basis Universal is essential

Our optimizer converts textures into KTX2 containers using two distinct codecs:
- **UASTC (for Normal & Roughness maps):** Fixed-rate 8 bits per texel. It transcodes directly on the GPU into desktop BC7 or mobile ASTC formats. It prevents blocky compression artifacts on normal maps that cause lighting seams.
- **ETC1S (for BaseColor & Emissive maps):** Uses an indexed codebook format at roughly 4 bits per texel. It drops VRAM consumption by 75% compared to raw bitmaps and compresses to tiny download sizes.

### Why meshopt instead of Draco?

Draco achieves slightly higher compression on disk, but requires a heavy WebAssembly decompression library. Decompressing Draco meshes on the fly causes visible frame stutters whenever a new unit spawns during a match.

`EXT_meshopt_compression` decompresses almost instantaneously in browser worker threads, supports random-access memory streaming, and preserves vertex cache locality for faster GPU rasterisation.

### How Vertex Animation Textures (VAT) work

Instead of passing a skeletal hierarchy to the GPU:
1. At build time, Blender samples the exact 3D position and normal vector of every vertex in the mesh on every frame of each animation clip.
2. These positions are normalized and stored as pixels in a 16-bit texture image. Row $Y$ corresponds to the animation frame; column $X$ corresponds to the vertex ID.
3. At runtime, the Three.js vertex shader reads the current animation time, samples the texture row, and displaces the static vertex to its animated position directly on the GPU.
4. Hundreds of units can share a single `InstancedMesh` draw call, completely eliminating CPU animation overhead.

---

## Advanced frontier: Reference-to-Video with Seedance for complex motion & VFX

While the core automated toolkit uses procedural gait equations and Mixamo retargeting for standard locomotion and attacks, complex creature behaviours often demand bespoke choreography and integrated visual effects that stock motion libraries lack.

To push beyond stock mocap, we are testing an experimental workflow combining Blender 3D proxy blocking with ByteDance's **Seedance** video diffusion model:

```
[ 3D Proxy / Viewport Previs ] ──► [ Playblast Reference Video ]
                                               │
                                               ▼
                                 [ Seedance R2V Diffusion ]
                                    (Guided by 3D timing)
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
            [ Motion Delta Tracking ]                    [ Synchronised VFX ]
         (Pose estimation to keyframes)               (Impact sprites & shaders)
                         │                                           │
                         ▼                                           ▼
            [ Blender Armature Action ]                  [ Engine Particle / Mesh ]
```

### How the Seedance workflow operates

1. **3D Previs & Proxy Blocking:** An artist or AI agent animates the model or simple proxy geometry inside Blender with rough keyframes. This sets the camera angle, timing windows, character trajectory, and physical limits, solving the typical "drifting camera" and proportion warping common in text-only video generation.
2. **Playblast Reference Video:** Blender exports a lightweight viewport render (clay or shaded) of the action.
3. **Reference-to-Video (R2V) Diffusion:** The playblast is fed into Seedance alongside detailed character and environmental prompt instructions. Guided by the 3D playblast, Seedance diffuses fluid anatomical weight shifts, complex multi-limb coordination, momentum recoils, and atmospheric interactions.
4. **Motion Retargeting:** Pose tracking extracts joint rotations from the generated video frames, mapping dynamic weight transfers and secondary physics back onto the Blender armature.
5. **VFX & Impact Extraction:** Character-aligned visual effects (such as water splashes, magical halos, and ground fracture shockwaves) are extracted as alpha-masked sprite sheets or billboard particles synchronised with the combat hit window.

> **Note:** This is an advanced frontier workflow currently in active experimental R&D. It is not yet bundled as a default one-click CLI command, but represents our forward roadmap for authoring high-fidelity, organic fantasy combat animations.

---

## Integrating custom units into Shards of Stone

Once your model is generated, dropping it into a custom map or mod is straightforward:

1. Copy the optimized GLB into your mod package directory:
   ```
   my-custom-map.sosmod/
     assets/
       models/neutral/units/reef_crab/character.glb
   ```
2. If using VAT rendering, place the baked files under `assets/models/baked/`:
   ```
   my-custom-map.sosmod/
     assets/
       models/baked/reef_crab_mesh.glb
       models/baked/reef_crab_vat_pos.ktx2
       models/baked/reef_crab_vat_norm.ktx2
       models/baked/reef_crab_vat.json
   ```
3. Reference the unit in your custom unit definitions (`customUnits.json`):
   ```json
   {
     "id": "custom_reef_crab",
     "name": "Reef Crab",
     "race": "neutral",
     "modelId": "reef_crab",
     "stats": {
       "hp": 450,
       "damage": 22,
       "armor": 5,
       "speed": 2.4
     }
   }
   ```
   **Important rule:** A 3D model without a matching `modelId` on its unit definition will not render. The engine will fall back to drawing the 2D billboard sprite.

---

## Contributing & open source license

This toolkit is licensed under the [MIT License](LICENSE). Contributions, new procedural body plan templates, and retargeting presets are welcome.

- Issues and discussions: [GitHub Issues](https://github.com/AdamJSturrock/shards-asset-toolkit/issues)
- Official game site: [shardsofstone.com](https://www.shardsofstone.com)
- Modding developer documentation: [docs.shardsofstone.com](https://www.shardsofstone.com/docs)
