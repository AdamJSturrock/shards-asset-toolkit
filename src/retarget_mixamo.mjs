// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Humanoid Mixamo retargeting via headless Blender (scripts/blender/retarget_mixamo.py).
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { runBlender, scriptDoc } from './blender.mjs';

const SCRIPT = 'scripts/blender/retarget_mixamo.py';

const HELP = `shards-asset retarget-mixamo: retarget Mixamo FBX clips onto a 24-joint Meshy-named humanoid

Usage:
  shards-asset retarget-mixamo --target rigged.glb --clip idle=mocap/idle.fbx --clip slash=mocap/slash.fbx \\
    --loop idle [--mirror slash] [--leg-align 0.5] --out candidates.glb

Options:
  --target <glb>        skinned humanoid (Meshy auto-rig, rig-humanoid or rig-fixup output)
  --clip name=fbx       repeatable; Mixamo FBX downloaded "without skin"
  --loop <name>         repeatable; keep that clip in place (horizontal root drift removed)
  --mirror <name>       repeatable; mirror the clip left-right (a right-handed slash for a
                        unit that holds its weapon in the left hand)
  --leg-align <0..1>    0 keeps the character's own leg stance, 1 copies Mixamo's (default 0.5)
  --keep-existing       also export the GLB's existing actions
  --out <glb>           candidate clips; pick and trim them with polish-clips

Known issue (weapon grip): Mixamo's rest fist is palm-down with the blade
pointing forward. T-pose concept art with the weapon drawn upright gives a
thumb-up fist, so retargeted clips carry a ~90 degree roll on the held weapon.
A grip normalise at rest is being built; until then check sword clips in
clip-review and correct the roll by hand if needed.

--- ${SCRIPT} ---
`;

export async function runRetargetMixamo(args) {
  const { values } = parseArgs({
    args,
    options: {
      target: { type: 'string' },
      clip: { type: 'string', multiple: true },
      loop: { type: 'string', multiple: true },
      mirror: { type: 'string', multiple: true },
      'leg-align': { type: 'string', default: '0.5' },
      'keep-existing': { type: 'boolean', default: false },
      out: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help || !values.target || !values.clip) {
    console.log(HELP + scriptDoc(SCRIPT));
    if (!values.help) process.exit(1);
    return;
  }

  const targetPath = path.resolve(process.cwd(), values.target);
  const outputPath = path.resolve(process.cwd(), values.out || './output/retargeted.glb');
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const blenderArgs = ['--target', targetPath, '--leg-align', values['leg-align'], '--out', outputPath];
  for (const c of values.clip) blenderArgs.push('--clip', c);
  for (const l of values.loop || []) blenderArgs.push('--loop', l);
  for (const m of values.mirror || []) blenderArgs.push('--mirror', m);
  if (values['keep-existing']) blenderArgs.push('--keep-existing');

  runBlender(SCRIPT, blenderArgs, { label: 'Retarget' });
  console.log(`[Retarget] Candidate clips written to: ${outputPath}`);
}
