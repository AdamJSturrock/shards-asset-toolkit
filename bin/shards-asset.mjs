#!/usr/bin/env node

/**
 * Shards of Stone 3D Asset Toolkit CLI
 * 
 * Orchestrates the complete generative 3D asset pipeline:
 * Concept (Gemini) -> Mesh (Meshy) -> Rigging (Blender) -> Optimize (glTF-Transform) -> VAT
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import dotenv from 'dotenv';

// Load .env from current directory or toolkit root
dotenv.config();

const USAGE = `
Shards of Stone 3D Asset Toolkit (@shardsofstone/asset-toolkit)

Usage:
  shards-asset <command> [options]

Commands:
  concept           Generate a clean 2D concept image with stripped atmospheric effects
  mesh              Convert 2D concept image to a 3D textured GLB via Meshy API
  rig-critter       Procedurally rig and animate non-humanoid creatures in headless Blender
  retarget-mixamo   Retarget Mixamo FBX motion clips to humanoid GLB models in Blender
  optimize          Transcode textures to KTX2 and compress geometry with meshopt
  bake-vat          Bake skeletal animation into Vertex Animation Textures (VAT)
  estimate-cost     Calculate API credits and token costs for a proposed generation
  pipeline          Execute the full end-to-end pipeline from a JSON configuration file

Options:
  --help, -h        Show this help message
  --version, -v     Show version information
`;

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === '--help' || command === '-h') {
    console.log(USAGE);
    process.exit(0);
  }

  if (command === '--version' || command === '-v') {
    const pkg = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf-8'));
    console.log(`v${pkg.version}`);
    process.exit(0);
  }

  const subArgs = args.slice(1);

  switch (command) {
    case 'concept': {
      const { runConcept } = await import('../src/concept.mjs');
      await runConcept(subArgs);
      break;
    }
    case 'mesh': {
      const { runMesh } = await import('../src/mesh.mjs');
      await runMesh(subArgs);
      break;
    }
    case 'rig-critter': {
      const { runRigCritter } = await import('../src/rig_critter.mjs');
      await runRigCritter(subArgs);
      break;
    }
    case 'retarget-mixamo': {
      const { runRetargetMixamo } = await import('../src/retarget_mixamo.mjs');
      await runRetargetMixamo(subArgs);
      break;
    }
    case 'optimize': {
      const { runOptimize } = await import('../src/optimize.mjs');
      await runOptimize(subArgs);
      break;
    }
    case 'bake-vat': {
      const { runBakeVat } = await import('../src/vat.mjs');
      await runBakeVat(subArgs);
      break;
    }
    case 'estimate-cost': {
      const { runEstimateCost } = await import('../src/pricing.mjs');
      await runEstimateCost(subArgs);
      break;
    }
    case 'pipeline': {
      const { runPipeline } = await import('../src/pipeline.mjs');
      await runPipeline(subArgs);
      break;
    }
    default:
      console.error(`Unknown command: ${command}`);
      console.log(USAGE);
      process.exit(1);
  }
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  if (process.env.DEBUG) {
    console.error(err.stack);
  }
  process.exit(1);
});
