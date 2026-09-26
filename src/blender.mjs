// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Shared helpers for running headless Blender and plain Python scripts, and
 * the table of CLI subcommands that forward their arguments straight to one
 * Blender script (scripts/blender/*.py).
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const TOOLKIT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function findBlender() {
  if (process.env.BLENDER_PATH && fs.existsSync(process.env.BLENDER_PATH)) {
    return process.env.BLENDER_PATH;
  }
  const standardPaths = [
    '/Applications/Blender.app/Contents/MacOS/Blender',
    '/usr/bin/blender',
    '/usr/local/bin/blender',
    '/opt/homebrew/bin/blender',
    'C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe',
    'C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe',
  ];
  for (const p of standardPaths) {
    if (fs.existsSync(p)) return p;
  }
  try {
    const test = spawnSync('blender', ['--version'], { encoding: 'utf-8' });
    if (test.status === 0) return 'blender';
  } catch {
    // not on PATH
  }
  return null;
}

export function requireBlender() {
  const bin = findBlender();
  if (!bin) {
    console.error('Error: Blender executable not found.');
    console.error('Install Blender 4.2+ and set BLENDER_PATH in your .env, or put "blender" on your PATH.');
    process.exit(1);
  }
  return bin;
}

/** Run a toolkit script inside headless Blender. Exits the process on failure. */
export function runBlender(scriptRel, scriptArgs = [], { label = 'Blender' } = {}) {
  const bin = requireBlender();
  const script = path.join(TOOLKIT_ROOT, scriptRel);
  const args = ['-b', '--factory-startup', '--python-exit-code', '1', '--python', script, '--', ...scriptArgs];
  console.log(`[${label}] ${path.basename(bin)} -b --python ${scriptRel} -- ${scriptArgs.join(' ')}`);
  const res = spawnSync(bin, args, { stdio: 'inherit' });
  if (res.status !== 0) {
    console.error(`[${label}] Blender exited with code ${res.status}`);
    process.exit(res.status || 1);
  }
}

/** Run a plain Python 3 script (numpy / scipy / Pillow, no Blender). */
export function runPython(scriptRel, scriptArgs = [], { label = 'Python' } = {}) {
  const py = process.env.PYTHON || 'python3';
  const script = path.join(TOOLKIT_ROOT, scriptRel);
  console.log(`[${label}] ${py} ${scriptRel} ${scriptArgs.join(' ')}`);
  const res = spawnSync(py, [script, ...scriptArgs], { stdio: 'inherit', env: process.env });
  if (res.status !== 0) {
    console.error(`[${label}] ${scriptRel} exited with code ${res.status}`);
    process.exit(res.status || 1);
  }
}

/** The module docstring of a Python script, for --help (no Blender needed). */
export function scriptDoc(scriptRel) {
  const src = fs.readFileSync(path.join(TOOLKIT_ROOT, scriptRel), 'utf-8');
  const m = src.match(/^(?:#[^\n]*\n|\s*\n)*("""|''')([\s\S]*?)\1/);
  return m ? m[2].trim() : '(no docstring)';
}

/**
 * Subcommands that forward every argument after the command name to one
 * Blender script. `--help` prints the script's own docstring.
 */
export const BLENDER_COMMANDS = {
  'rig-humanoid': {
    script: 'scripts/blender/rig_humanoid.py',
    summary: "Rig a static Meshy humanoid on Meshy's 24 joints in Blender (for meshes Meshy's auto-rig refuses)",
    usage: '--spec examples/anim-authoring/shellback_cutthroat.json --out output/anim-authoring/shellback_cutthroat/rig-v3/character.glb',
  },
  'rig-fixup': {
    script: 'scripts/blender/rig_fixup.py',
    summary: 'Fix a Meshy auto-rig: weapon 100% to the hand, tail chains, skirt/cloak weights, attach a shield prop',
    usage: '--spec examples/anim-authoring/dwarf_axe_shield.json --out output/anim-authoring/dwarf_axe_shield/rig-v1/rig.glb',
  },
  'rig-mount': {
    script: 'scripts/blender/rig_mount.py',
    summary: 'Rig a rider-on-a-mount mesh as one skinned model: quadruped mount + 24-joint rider',
    usage: '--spec examples/anim-authoring/mounted_knight.json --out output/anim-authoring/mounted_knight/rig-v1/character.glb',
  },
  'mount-clips': {
    script: 'scripts/blender/mount_clips.py',
    summary: 'Author mount clips (procedural horse + retargeted rider) into one GLB',
    usage: '--spec examples/anim-authoring/mounted_knight.json --out output/anim-authoring/mounted_knight/v1/character.glb',
  },
  'repair-head-weights': {
    script: 'scripts/blender/repair_head_weights.py',
    summary: 'Strip shoulder/arm influence that Meshy auto-rigs leak into the head',
    usage: '--input <rigged character.glb> --out output/anim-authoring/<unit>/rig-v1/character.glb',
  },
  'polish-clips': {
    script: 'scripts/blender/polish_clips.py',
    summary: 'Turn retargeted candidate clips into a final clip set from a JSON spec (trim, speed, damping, props)',
    usage: '--spec examples/anim-authoring/shellback_cutthroat.json --out output/anim-authoring/shellback_cutthroat/v3/character.glb',
  },
  'clip-review': {
    script: 'scripts/blender/clip_review.py',
    summary: 'Render sampled frames of every clip (game 3/4, side, front views) for review',
    usage: '--input <animated.glb> --output-dir output/review/<unit> --samples 8',
  },
  'rig-pose-check': {
    script: 'scripts/blender/rig_pose_check.py',
    summary: 'Render a rig in stress-test poses (arms raised, crouch, twist) to expose bad weights',
    usage: '--input <rigged.glb> --out output/review/<unit>/rigcheck.png',
  },
  'weight-review': {
    script: 'scripts/blender/meshy_rig_weight_review.py',
    summary: 'Paint each vertex by its dominant bone group to see how a Meshy auto-rig skinned the weapon and hands',
    usage: '<targets.json> [size]',
  },
};

export function printBlenderCommandHelp(name) {
  const c = BLENDER_COMMANDS[name];
  console.log(`shards-asset ${name}: ${c.summary}\n`);
  console.log(`Usage:\n  shards-asset ${name} ${c.usage}\n`);
  console.log(`Runs: blender -b --factory-startup --python-exit-code 1 --python ${c.script} -- <args>\n`);
  console.log(`--- ${c.script} ---\n${scriptDoc(c.script)}`);
}

export function runBlenderCommand(name, args) {
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    printBlenderCommandHelp(name);
    return;
  }
  runBlender(BLENDER_COMMANDS[name].script, args, { label: name });
}
