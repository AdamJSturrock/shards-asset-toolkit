/**
 * Vertex Animation Textures (VAT) baking runner.
 * Spawns headless Blender to bake per-frame vertex offsets into 16-bit texture maps.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { findBlender } from './rig_critter.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export async function runBakeVat(args) {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: 'string' },
      'output-dir': { type: 'string', default: './output/baked' },
      'unit-id': { type: 'string' },
      'target-verts': { type: 'string', default: '4000' }
    }
  });

  if (!values.input) {
    console.error('Error: --input <path-to-rigged.glb> is required');
    process.exit(1);
  }

  const blenderBin = findBlender();
  if (!blenderBin) {
    console.error('Error: Blender executable not found.');
    process.exit(1);
  }

  const inputPath = path.resolve(process.cwd(), values.input);
  const outputDir = path.resolve(process.cwd(), values['output-dir']);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const unitId = values['unit-id'] || path.basename(inputPath, '.glb');
  const targetVerts = parseInt(values['target-verts'], 10) || 4000;

  console.log(`[Bake-VAT] Baking Vertex Animation Textures for unit: ${unitId}...`);
  console.log(`[Bake-VAT] Target decimated vertices: ${targetVerts}`);
  console.log(`[Bake-VAT] Output directory: ${outputDir}`);

  const scriptPath = path.resolve(__dirname, '../scripts/blender/vat_bake.py');

  const blenderArgs = [
    '-b',
    '--factory-startup',
    '--python-exit-code', '1',
    '--python', scriptPath,
    '--',
    '--input', inputPath,
    '--unit-id', unitId,
    '--target-verts', String(targetVerts),
    '--output-dir', outputDir
  ];

  const result = spawnSync(blenderBin, blenderArgs, {
    stdio: 'inherit',
    encoding: 'utf-8'
  });

  if (result.status !== 0) {
    console.error(`[Bake-VAT] Blender VAT bake failed with exit code: ${result.status}`);
    process.exit(1);
  }

  console.log(`[Bake-VAT] VAT bundle baked successfully!`);
  console.log(`  Mesh:     ${path.join(outputDir, `${unitId}_mesh.glb`)}`);
  console.log(`  Position: ${path.join(outputDir, `${unitId}_vat_pos.png`)}`);
  console.log(`  Normals:  ${path.join(outputDir, `${unitId}_vat_norm.png`)}`);
  console.log(`  Metadata: ${path.join(outputDir, `${unitId}_vat.json`)}`);
}
