#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors

/**
 * Shards of Stone 3D Asset Toolkit CLI
 *
 * Concept (Gemini) -> Mesh (Meshy) -> Rig (Meshy auto-rig or Blender) ->
 * Clips (Mixamo retarget + polish, or procedural critter gaits) ->
 * Dye mask -> Optimize (glTF-Transform). VAT baking is experimental.
 */

import fs from 'node:fs';
import dotenv from 'dotenv';
import { BLENDER_COMMANDS, runBlenderCommand } from '../src/blender.mjs';

dotenv.config({ quiet: true });

const blenderLines = Object.entries(BLENDER_COMMANDS)
  .map(([name, c]) => `  ${name.padEnd(20)}${c.summary}`)
  .join('\n');

const USAGE = `
Shards of Stone 3D Asset Toolkit (@shardsofstone/asset-toolkit)

Usage:
  shards-asset <command> [options]
  shards-asset <command> --help

Generate:
  concept             Generate a clean 2D concept image (Gemini)
  mesh                Convert a concept image into a textured GLB (Meshy image-to-3D, 30 credits)
  meshy-rig           Meshy humanoid auto-rig of a mesh task (5 credits, no animations)

Rig and animate (headless Blender):
  rig-critter         Rig and animate multi-legged creatures from a profile (prep, fit, rig, review)
  retarget-mixamo     Retarget Mixamo FBX clips onto a 24-joint humanoid
${blenderLines}

Finish:
  dye-mask            Add, inspect or verify the player-colour dye mask (apply | bbox | verify)
  optimize            Transcode textures to KTX2 and compress geometry with meshopt
  bake-vat            Bake animation into Vertex Animation Textures (EXPERIMENTAL)

Other:
  estimate-cost       Meshy credits and Gemini image cost for a batch of units
  pipeline            Run the stages for one unit from a JSON config

Options:
  --help, -h          Show this help message
  --version, -v       Show version information
`;

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === '--help' || command === '-h' || command === 'help') {
    console.log(USAGE);
    process.exit(0);
  }

  if (command === '--version' || command === '-v') {
    const pkg = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf-8'));
    console.log(`v${pkg.version}`);
    process.exit(0);
  }

  const subArgs = args.slice(1);

  if (BLENDER_COMMANDS[command]) {
    runBlenderCommand(command, subArgs);
    return;
  }

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
    case 'meshy-rig': {
      const { runMeshyRig } = await import('../src/meshy_rig.mjs');
      await runMeshyRig(subArgs);
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
    case 'dye-mask': {
      const { runDyeMask } = await import('../src/dye_mask.mjs');
      await runDyeMask(subArgs);
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
