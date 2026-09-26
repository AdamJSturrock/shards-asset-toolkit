// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Non-humanoid rigging: the critter-authoring pipeline in scripts/critter/.
 * Profile-driven (scripts/critter/profiles.json, or $CRITTER_PROFILES):
 * prep -> fit (traced limbs + label checks) -> rig (armature, per-limb heat
 * weights, procedural clips) -> review.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { runBlender, runPython, TOOLKIT_ROOT, findBlender } from './blender.mjs';

// Kept for backwards compatibility: vat.mjs and retarget_mixamo.mjs used to import it from here.
export { findBlender };

const HELP = `shards-asset rig-critter: rig and animate a multi-legged creature (headless Blender + Python)

Usage:
  shards-asset rig-critter <step> <critter> [options]

<critter> is a profile name in scripts/critter/profiles.json (or the file in
$CRITTER_PROFILES). Body plans: arachnid, crustacean, scorpion, drake, sprawl
(traced by fit_arthropod.py); quadruped, bird, frog take hand-placed joints.
Working files go to ./output/critter-authoring/<critter>/ (or $CRITTER_AUTHORING_OUT).

Steps:
  standin --kind spider|scorpion|drake|salamander --out <glb>
                  build a procedural stand-in mesh (no Meshy credits); the
                  *_standin profiles point at output/standins/<kind>.glb
  prep <critter> [--input <glb>]
                  rig-space renders, verts/faces, shape diameter (Blender)
  fit <critter> [--allow-fail]
                  trace every limb from its tip, derive the joints, write
                  per-vertex limb labels, run the label checks, then render
                  labels.png and the fit.png landmark grid. A failed label
                  check stops here: LOOK at labels.png before rigging.
  rig <critter> <version> [--input <glb>] [--set k=v,...]
                  build the armature, skin with per-limb heat weights,
                  author idle/walk/attack/death/special clips, export
                  <work>/<critter>/<version>/character.glb (Blender)
  review <critter> <version> [--no-video]
                  validation.json, weights.png, per-clip renders, preview.mp4
                  (needs ffmpeg), plus a clip contact sheet (Pillow)
  check <critter> <version>
                  reopen the saved .blend and prove every action moves the mesh
  add-limb <critter> --copy leg2.L,leg2.R --out <glb> [--mirror 1] [--yaw deg] [--move dx,dy,dz] [--grow n]
                  copy a labelled limb to replace one Meshy dropped
  all <critter> <version>
                  rig + review

Requirements: Blender 4.2+, python3 with numpy, scipy and Pillow; ffmpeg for
the review video.

Example (no API keys needed):
  shards-asset rig-critter standin --kind spider --out output/standins/spider.glb
  shards-asset rig-critter prep spider_standin
  shards-asset rig-critter fit spider_standin
  shards-asset rig-critter all spider_standin v1`;

function loadProfiles() {
  const file = process.env.CRITTER_PROFILES || path.join(TOOLKIT_ROOT, 'scripts', 'critter', 'profiles.json');
  return JSON.parse(fs.readFileSync(file, 'utf-8'));
}

function workDir(...parts) {
  const base = process.env.CRITTER_AUTHORING_OUT || path.join(process.cwd(), 'output', 'critter-authoring');
  return path.join(base.replace(/^~(?=$|\/)/, process.env.HOME || '~'), ...parts);
}

function takeOpt(args, name) {
  const i = args.indexOf(name);
  if (i < 0) return undefined;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}

function takeFlag(args, name) {
  const i = args.indexOf(name);
  if (i < 0) return false;
  args.splice(i, 1);
  return true;
}

export async function runRigCritter(argv) {
  const args = [...argv];
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(HELP);
    return;
  }
  // Backwards compatibility with the 1.0 flags (--input/--template/--output).
  if (args[0].startsWith('--')) {
    console.error('rig-critter now takes a step and a profile name, e.g. "rig-critter all reef_crab v1".');
    console.error('Run "shards-asset rig-critter --help" for the steps.');
    process.exit(1);
  }
  const step = args.shift();

  if (step === 'standin') {
    const kind = takeOpt(args, '--kind');
    const out = takeOpt(args, '--out') || `output/standins/${kind}.glb`;
    if (!kind) throw new Error('standin needs --kind spider|scorpion|drake|salamander');
    fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
    runBlender('scripts/critter/make_standin.py', ['--kind', kind, '--out', out, ...args], { label: 'Standin' });
    return;
  }

  const name = args.shift();
  if (!name) throw new Error(`${step} needs a critter profile name`);
  const profiles = loadProfiles();
  const prof = profiles[name];
  if (!prof) throw new Error(`no profile "${name}" (have: ${Object.keys(profiles).join(', ')})`);

  const sourceOf = (override) => {
    if (override) return override;
    if (!prof.source) throw new Error(`profile "${name}" has no "source"; pass --input <glb>`);
    const s = prof.source.replace(/^~(?=$|\/)/, process.env.HOME || '~');
    return path.isAbsolute(s) ? s : path.resolve(process.cwd(), s);
  };

  switch (step) {
    case 'prep': {
      const input = sourceOf(takeOpt(args, '--input'));
      runBlender('scripts/critter/prep_critter.py', [
        '--input', input, '--out-dir', workDir(name, 'prep'),
        '--yaw', String(prof.yaw_deg ?? 0), '--pitch', String(prof.pitch_deg ?? 0),
        '--center-x', String(prof.center_x ?? 'bbox'), ...args,
      ], { label: 'Prep' });
      break;
    }
    case 'fit': {
      const allowFail = takeFlag(args, '--allow-fail');
      const py = process.env.PYTHON || 'python3';
      const fitArgs = [path.join(TOOLKIT_ROOT, 'scripts/critter/fit_arthropod.py'), name, '--write', ...args];
      if (allowFail) fitArgs.push('--allow-fail');
      console.log(`[Fit] ${py} scripts/critter/fit_arthropod.py ${name} --write`);
      const fit = spawnSync(py, fitArgs, { stdio: 'inherit', env: process.env });
      // Render the labels EVEN when a check fails: that is when they need looking at.
      runPython('scripts/critter/render_labels.py', [name], { label: 'Labels' });
      runPython('scripts/critter/fit_landmarks.py', [name], { label: 'Landmarks' });
      if (fit.status !== 0) {
        console.error(`[Fit] fit failed (exit ${fit.status}): look at ${workDir(name, 'labels.png')} and prep/fit-report.json`);
        process.exit(fit.status || 1);
      }
      break;
    }
    case 'rig':
    case 'review':
    case 'all':
    case 'check': {
      const version = args.shift();
      if (!version) throw new Error(`${step} needs a version label, e.g. v1`);
      const dir = workDir(name, version);
      fs.mkdirSync(dir, { recursive: true });
      if (step === 'check') {
        runBlender('scripts/critter/check_master.py', ['--blend', path.join(dir, `${name}.blend`)], { label: 'Check' });
        break;
      }
      if (step === 'rig' || step === 'all') {
        const input = takeOpt(args, '--input');
        const set = takeOpt(args, '--set');
        const rigArgs = ['--critter', name, '--out-dir', dir];
        if (input) rigArgs.push('--input', input);
        if (set) rigArgs.push('--set', set);
        runBlender('scripts/critter/rig_critter.py', rigArgs, { label: 'Rig' });
      }
      if (step === 'review' || step === 'all') {
        const noVideo = takeFlag(args, '--no-video');
        const glb = path.join(dir, 'character.glb');
        runBlender('scripts/critter/review_rig.py', ['--input', glb, '--out-dir', path.join(dir, 'review'), '--video', noVideo ? '0' : '1'], { label: 'Review' });
        runBlender('scripts/blender/clip_review.py', ['--input', glb, '--output-dir', path.join(dir, 'sheet'), '--samples', '8', '--resolution', '256'], { label: 'Clip review' });
        runPython('scripts/clip_review_sheet.py', ['--review-dir', path.join(dir, 'sheet'), '--out', path.join(dir, 'contact_sheet.png'), '--gameplay-px', '48', '--cell', '160'], { label: 'Sheet' });
        console.log(`[Review] Look at ${path.join(dir, 'contact_sheet.png')} and ${path.join(dir, 'review')}: numbers are evidence, not approval.`);
      }
      break;
    }
    case 'add-limb': {
      runBlender('scripts/critter/add_limb.py', ['--critter', name, ...args], { label: 'Add limb' });
      break;
    }
    default:
      console.error(`Unknown rig-critter step: ${step}`);
      console.log(HELP);
      process.exit(1);
  }
}
