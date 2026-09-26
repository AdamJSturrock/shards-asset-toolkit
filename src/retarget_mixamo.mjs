/**
 * Humanoid Mixamo retargeting module via headless Blender.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { findBlender } from './rig_critter.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export async function runRetargetMixamo(args) {
  const { values } = parseArgs({
    args,
    options: {
      target: { type: 'string' },
      clip: { type: 'string', multiple: true },
      loop: { type: 'string', multiple: true },
      'leg-align': { type: 'string', default: '0.5' },
      out: { type: 'string' },
    }
  });

  if (!values.target) {
    console.error('Error: --target <path-to-humanoid.glb> is required');
    process.exit(1);
  }

  const blenderBin = findBlender();
  if (!blenderBin) {
    console.error('Error: Blender executable not found.');
    process.exit(1);
  }

  const targetPath = path.resolve(process.cwd(), values.target);
  const outputPath = path.resolve(process.cwd(), values.out || './output/retargeted.glb');
  const outDir = path.dirname(outputPath);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const scriptPath = path.resolve(__dirname, '../scripts/blender/retarget_mixamo.py');

  const blenderArgs = [
    '-b',
    '--factory-startup',
    '--python-exit-code', '1',
    '--python', scriptPath,
    '--',
    '--target', targetPath,
    '--leg-align', values['leg-align'] || '0.5',
    '--out', outputPath
  ];

  if (values.clip) {
    for (const c of values.clip) {
      blenderArgs.push('--clip', c);
    }
  }

  if (values.loop) {
    for (const l of values.loop) {
      blenderArgs.push('--loop', l);
    }
  }

  console.log(`[Retarget] Retargeting Mixamo clips to ${targetPath}...`);
  console.log(`[Retarget] Leg align ratio: ${values['leg-align']}`);

  const result = spawnSync(blenderBin, blenderArgs, {
    stdio: 'inherit',
    encoding: 'utf-8'
  });

  if (result.status !== 0) {
    console.error(`[Retarget] Blender failed with exit code: ${result.status}`);
    process.exit(1);
  }

  console.log(`[Retarget] Clips retargeted successfully to: ${outputPath}`);
}
