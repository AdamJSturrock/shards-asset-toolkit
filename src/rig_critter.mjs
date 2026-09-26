/**
 * Non-humanoid procedural rigging runner via headless Blender.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function findBlender() {
  if (process.env.BLENDER_PATH && fs.existsSync(process.env.BLENDER_PATH)) {
    return process.env.BLENDER_PATH;
  }

  const standardPaths = [
    '/Applications/Blender.app/Contents/MacOS/Blender',
    '/usr/bin/blender',
    '/usr/local/bin/blender',
    'C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe',
    'C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe',
  ];

  for (const p of standardPaths) {
    if (fs.existsSync(p)) return p;
  }

  // Fallback to searching PATH
  try {
    const test = spawnSync('blender', ['--version'], { encoding: 'utf-8' });
    if (test.status === 0) return 'blender';
  } catch (e) {
    // not in PATH
  }

  return null;
}

export async function runRigCritter(args) {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: 'string' },
      template: { type: 'string', default: 'crustacean' },
      output: { type: 'string' },
    }
  });

  if (!values.input) {
    console.error('Error: --input <path-to-mesh.glb> is required');
    process.exit(1);
  }

  const blenderBin = findBlender();
  if (!blenderBin) {
    console.error('Error: Blender executable not found.');
    console.error('Please install Blender 4.2+ and set BLENDER_PATH in your .env or ensure "blender" is in PATH.');
    process.exit(1);
  }

  const inputPath = path.resolve(process.cwd(), values.input);
  const outputPath = path.resolve(process.cwd(), values.output || './output/rigged.glb');
  const outDir = path.dirname(outputPath);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const scriptPath = path.resolve(__dirname, '../scripts/blender/fit_arthropod.py');

  console.log(`[Rig-Critter] Launching headless Blender with template: ${values.template}...`);
  console.log(`[Rig-Critter] Input:  ${inputPath}`);
  console.log(`[Rig-Critter] Output: ${outputPath}`);

  const blenderArgs = [
    '-b',
    '--factory-startup',
    '--python-exit-code', '1',
    '--python', scriptPath,
    '--',
    '--input', inputPath,
    '--template', values.template,
    '--output', outputPath
  ];

  const result = spawnSync(blenderBin, blenderArgs, {
    stdio: 'inherit',
    encoding: 'utf-8'
  });

  if (result.status !== 0) {
    console.error(`[Rig-Critter] Blender failed with exit code: ${result.status}`);
    process.exit(1);
  }

  console.log(`[Rig-Critter] Rigging and animation completed successfully!`);
}
