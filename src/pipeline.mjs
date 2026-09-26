// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * End-to-end runner from one JSON unit config. Each stage is skipped when its
 * output already exists, so a run can be resumed or a stage redone by
 * deleting its file. Spec-driven Blender steps (rig-humanoid, rig-fixup,
 * polish-clips) carry their own paths in their spec and are run on their own.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { runConcept } from './concept.mjs';
import { runMesh } from './mesh.mjs';
import { runMeshyRig } from './meshy_rig.mjs';
import { runRigCritter } from './rig_critter.mjs';
import { runRetargetMixamo } from './retarget_mixamo.mjs';
import { runDyeMask } from './dye_mask.mjs';
import { runOptimize } from './optimize.mjs';
import { runBakeVat } from './vat.mjs';
import { runEstimateCost } from './pricing.mjs';

const HELP = `shards-asset pipeline: run the stages for one unit from a JSON config

Usage:
  shards-asset pipeline --config examples/units/goblin_worker.json

Config fields:
  unitId, outputDir
  prompt | conceptImage       concept stage (tpose, keyColours pass through)
  polycount, aiModel, poseMode, symmetry   Meshy mesh stage
  type: "humanoid"            rig: riggedGlb (an existing rig, e.g. from rig-humanoid or
                              rig-fixup) or Meshy auto-rig (height); then Mixamo clips:
                              clips {name: fbx}, loops [...], mirror [...], legAlign
  type: "critter"             rig: profile (scripts/critter/profiles.json), version
  dyeMask: true               add the player-colour mask (magenta/cyan key colours)
  textureMax                  optimize stage
  vat: true                   EXPERIMENTAL VAT bake`;

export async function runPipeline(args) {
  const { values } = parseArgs({ args, options: { config: { type: 'string' }, help: { type: 'boolean', short: 'h' } } });
  if (values.help || !values.config) {
    console.log(HELP);
    if (!values.help) process.exit(1);
    return;
  }
  const configPath = path.resolve(process.cwd(), values.config);
  const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  const unitId = config.unitId || 'custom_unit';
  const outBase = path.resolve(process.cwd(), config.outputDir || `./output/${unitId}`);
  fs.mkdirSync(outBase, { recursive: true });

  const conceptPath = path.join(outBase, `${unitId}_concept.png`);
  const rawMeshPath = path.join(outBase, `${unitId}_raw.glb`);
  const riggedPath = path.join(outBase, `${unitId}_rigged.glb`);
  const animatedPath = path.join(outBase, `${unitId}_animated.glb`);
  const dyedPath = path.join(outBase, `${unitId}_dyed.glb`);
  const optimizedPath = path.join(outBase, `${unitId}_character.glb`);

  console.log(`\n=== Shards of Stone asset pipeline: ${unitId} ===\n`);
  await runEstimateCost(['--units', '1', '--candidates', '1', ...(config.type === 'critter' ? ['--no-meshy-rig'] : [])]);

  // 1. Concept
  if (!fs.existsSync(conceptPath)) {
    if (config.conceptImage) {
      fs.copyFileSync(path.resolve(process.cwd(), config.conceptImage), conceptPath);
    } else if (config.prompt) {
      console.log('\n>>> Stage 1: concept (Gemini)');
      const a = ['--prompt', config.prompt, '--output', conceptPath];
      if (config.tpose) a.push('--tpose');
      if (config.keyColours || config.dyeMask) a.push('--key-colours');
      await runConcept(a);
    }
  }

  // 2. Mesh
  if (!fs.existsSync(rawMeshPath) && fs.existsSync(conceptPath)) {
    console.log('\n>>> Stage 2: mesh (Meshy image-to-3D)');
    const a = ['--input', conceptPath, '--output', rawMeshPath, '--polycount', String(config.polycount || 30000)];
    if (config.aiModel) a.push('--ai-model', config.aiModel);
    if (config.poseMode) a.push('--pose-mode', config.poseMode);
    if (config.symmetry) a.push('--symmetry', config.symmetry);
    await runMesh(a);
  }

  // 3. Rig and animate
  if (config.type === 'critter') {
    if (!config.profile) throw new Error('a critter config needs "profile" (a name in scripts/critter/profiles.json)');
    const version = config.version || 'v1';
    const critterGlb = path.join(process.env.CRITTER_AUTHORING_OUT || path.join(process.cwd(), 'output', 'critter-authoring'), config.profile, version, 'character.glb');
    if (!fs.existsSync(animatedPath)) {
      console.log('\n>>> Stage 3: critter rig (prep, fit, rig, review)');
      await runRigCritter(['prep', config.profile, '--input', rawMeshPath]);
      await runRigCritter(['fit', config.profile]);
      await runRigCritter(['all', config.profile, version, '--input', rawMeshPath]);
      fs.copyFileSync(critterGlb, animatedPath);
    }
  } else {
    if (!fs.existsSync(riggedPath)) {
      if (config.riggedGlb) {
        fs.copyFileSync(path.resolve(process.cwd(), config.riggedGlb), riggedPath);
      } else if (fs.existsSync(rawMeshPath)) {
        console.log('\n>>> Stage 3: Meshy humanoid auto-rig');
        await runMeshyRig(['--task-json', rawMeshPath.replace(/\.glb$/, '.meshy.json'), '--output', riggedPath,
          '--height', String(config.height || 1.7)]);
      }
    }
    if (!fs.existsSync(animatedPath) && fs.existsSync(riggedPath) && config.clips) {
      console.log('\n>>> Stage 4: Mixamo retarget');
      const a = ['--target', riggedPath, '--out', animatedPath, '--leg-align', String(config.legAlign ?? 0.5)];
      for (const [name, fbx] of Object.entries(config.clips)) a.push('--clip', `${name}=${fbx}`);
      for (const l of config.loops || []) a.push('--loop', l);
      for (const m of config.mirror || []) a.push('--mirror', m);
      await runRetargetMixamo(a);
      console.log('    These are candidate clips: trim and polish them with polish-clips, then look at clip-review.');
    }
  }

  // 5. Dye mask
  let finalGlb = animatedPath;
  if (config.dyeMask && fs.existsSync(animatedPath)) {
    if (!fs.existsSync(dyedPath)) {
      console.log('\n>>> Stage 5: player-colour dye mask');
      await runDyeMask(['apply', '--in', animatedPath, '--out', dyedPath, '--review', dyedPath.replace(/\.glb$/, '_review.png')]);
    }
    finalGlb = dyedPath;
  }

  // 6. Optimize
  if (!fs.existsSync(optimizedPath) && fs.existsSync(finalGlb)) {
    console.log('\n>>> Stage 6: optimize (KTX2 + meshopt)');
    await runOptimize(['--input', finalGlb, '--output', optimizedPath, '--texture-max', String(config.textureMax || 1024)]);
  }

  // 7. VAT (experimental)
  if (config.vat && fs.existsSync(finalGlb)) {
    console.log('\n>>> Stage 7: VAT bake (EXPERIMENTAL)');
    await runBakeVat(['--input', finalGlb, '--output-dir', path.join(outBase, 'baked'), '--unit-id', unitId]);
  }

  console.log(`\n=== Done. Outputs in ${outBase} ===`);
  console.log('Look at the renders (clip-review, rig-pose-check) before shipping: numbers are evidence, not approval.\n');
}
