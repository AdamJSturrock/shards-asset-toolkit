// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Vertex Animation Texture (VAT) baking via headless Blender. EXPERIMENTAL:
 * the Shards of Stone game renders units with standard GPU skinning plus
 * instancing today; it does not use this path.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { runBlender } from './blender.mjs';

const HELP = `shards-asset bake-vat: bake a skinned, animated GLB into a VAT bundle (EXPERIMENTAL)

Usage:
  shards-asset bake-vat --input animated.glb [--output-dir ./output/baked] [--unit-id name]
                        [--target-verts 4000] [--max-frames 256] [--fps 30] [--animations clips.glb]

Writes <unit>_mesh.glb (static mesh, UV2 vertex ids), <unit>_vat_pos.png and
<unit>_vat_norm.png (16-bit) and <unit>_vat.json (clip table).

Experimental: the Shards of Stone game currently draws units as skinned GLBs
with standard GPU skinning plus instancing, not VAT. The baker works, but no
frame-time or memory numbers have been measured for it in the game.`;

export async function runBakeVat(args) {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: 'string' },
      'output-dir': { type: 'string', default: './output/baked' },
      'unit-id': { type: 'string' },
      'target-verts': { type: 'string', default: '4000' },
      'max-frames': { type: 'string', default: '256' },
      fps: { type: 'string', default: '30' },
      animations: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help || !values.input) {
    console.log(HELP);
    if (!values.help) process.exit(1);
    return;
  }

  const inputPath = path.resolve(process.cwd(), values.input);
  const outputDir = path.resolve(process.cwd(), values['output-dir']);
  fs.mkdirSync(outputDir, { recursive: true });
  const unitId = values['unit-id'] || path.basename(inputPath, '.glb');

  console.log('[Bake-VAT] EXPERIMENTAL: the game does not render units through VAT today.');
  const blenderArgs = [
    '--input', inputPath,
    '--unit-id', unitId,
    '--output-dir', outputDir,
    '--target-verts', values['target-verts'],
    '--max-frames', values['max-frames'],
    '--fps', values.fps,
  ];
  if (values.animations) blenderArgs.push('--animations', path.resolve(process.cwd(), values.animations));
  runBlender('scripts/blender/vat_bake.py', blenderArgs, { label: 'Bake-VAT' });

  console.log('[Bake-VAT] VAT bundle written:');
  console.log(`  Mesh:     ${path.join(outputDir, `${unitId}_mesh.glb`)}`);
  console.log(`  Position: ${path.join(outputDir, `${unitId}_vat_pos.png`)}`);
  console.log(`  Normals:  ${path.join(outputDir, `${unitId}_vat_norm.png`)}`);
  console.log(`  Metadata: ${path.join(outputDir, `${unitId}_vat.json`)}`);
}
