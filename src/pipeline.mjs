/**
 * End-to-end pipeline runner from a single JSON configuration.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { runConcept } from './concept.mjs';
import { runMesh } from './mesh.mjs';
import { runRigCritter } from './rig_critter.mjs';
import { runRetargetMixamo } from './retarget_mixamo.mjs';
import { runOptimize } from './optimize.mjs';
import { runBakeVat } from './vat.mjs';
import { runEstimateCost } from './pricing.mjs';

export async function runPipeline(args) {
  const { values } = parseArgs({
    args,
    options: {
      config: { type: 'string' }
    }
  });

  if (!values.config) {
    console.error('Error: --config <path-to-config.json> is required');
    process.exit(1);
  }

  const configPath = path.resolve(process.cwd(), values.config);
  if (!fs.existsSync(configPath)) {
    console.error(`Error: Config file not found: ${configPath}`);
    process.exit(1);
  }

  const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  const unitId = config.unitId || 'custom_unit';
  const outBase = path.resolve(process.cwd(), config.outputDir || `./output/${unitId}`);

  console.log(`\n======================================================`);
  console.log(`Starting Shards of Stone 3D Asset Pipeline for: ${unitId}`);
  console.log(`======================================================\n`);

  fs.mkdirSync(outBase, { recursive: true });

  const conceptPath = path.join(outBase, `${unitId}_concept.png`);
  const rawMeshPath = path.join(outBase, `${unitId}_raw.glb`);
  const riggedMeshPath = path.join(outBase, `${unitId}_rigged.glb`);
  const optimizedMeshPath = path.join(outBase, `${unitId}_character.glb`);
  const bakedDir = path.join(outBase, 'baked');

  // Step 0: Cost estimate
  await runEstimateCost(['--units', '1', '--candidates', '3']);

  // Step 1: Concept
  if (config.prompt && !fs.existsSync(conceptPath)) {
    console.log(`\n>>> Stage 1: Generating Concept Art...`);
    await runConcept([
      '--prompt', config.prompt,
      '--output', conceptPath,
      '--strip-effects'
    ]);
  } else if (config.conceptImage && fs.existsSync(config.conceptImage)) {
    fs.copyFileSync(config.conceptImage, conceptPath);
  }

  // Step 2: Mesh
  if (!fs.existsSync(rawMeshPath) && fs.existsSync(conceptPath)) {
    console.log(`\n>>> Stage 2: 3D Mesh Synthesis with Meshy...`);
    await runMesh([
      '--input', conceptPath,
      '--output', rawMeshPath,
      '--polycount', String(config.polycount || 30000)
    ]);
  }

  // Step 3: Rigging
  if (!fs.existsSync(riggedMeshPath) && fs.existsSync(rawMeshPath)) {
    console.log(`\n>>> Stage 3: Rigging & Animation...`);
    if (config.type === 'humanoid' && config.clips) {
      const clipArgs = ['--target', rawMeshPath, '--out', riggedMeshPath];
      for (const [name, clipPath] of Object.entries(config.clips)) {
        clipArgs.push('--clip', `${name}=${clipPath}`);
      }
      if (config.loops) {
        for (const loopName of config.loops) {
          clipArgs.push('--loop', loopName);
        }
      }
      if (config.legAlign !== undefined) {
        clipArgs.push('--leg-align', String(config.legAlign));
      }
      await runRetargetMixamo(clipArgs);
    } else {
      await runRigCritter([
        '--input', rawMeshPath,
        '--template', config.template || 'crustacean',
        '--output', riggedMeshPath
      ]);
    }
  }

  // Step 4: Optimization
  if (!fs.existsSync(optimizedMeshPath) && fs.existsSync(riggedMeshPath)) {
    console.log(`\n>>> Stage 4: Web RTS Optimization (KTX2 & meshopt)...`);
    await runOptimize([
      '--input', riggedMeshPath,
      '--output', optimizedMeshPath,
      '--texture-max', String(config.textureMax || 1024)
    ]);
  }

  // Step 5: VAT
  if (config.vat && fs.existsSync(riggedMeshPath)) {
    console.log(`\n>>> Stage 5: Baking Vertex Animation Textures (VAT)...`);
    await runBakeVat([
      '--input', riggedMeshPath,
      '--output-dir', bakedDir,
      '--unit-id', unitId,
      '--target-verts', String(config.targetVerts || 4000)
    ]);
  }

  console.log(`\n======================================================`);
  console.log(`Pipeline Complete! Production assets ready at:`);
  console.log(`  Optimized GLB: ${optimizedMeshPath}`);
  if (config.vat) {
    console.log(`  VAT Bundle:    ${bakedDir}`);
  }
  console.log(`======================================================\n`);
}
