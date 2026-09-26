// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Meshy humanoid AUTO-RIG only (5 credits): POST /openapi/v1/rigging with the
 * image-to-3D task id, download the rigged GLB. No Meshy animations are
 * requested; clips come from Mixamo (retarget-mixamo).
 *
 * The rig is requested with `input_task_id` alone. Passing a texture image
 * here re-textures the mesh.
 */

import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';
import { meshyRequest, pollTask, download } from './mesh.mjs';

const API_ROOT = 'https://api.meshy.ai/openapi/v1';
export const MESHY_RIG_CREDITS = 5;

const HELP = `shards-asset meshy-rig: Meshy humanoid auto-rig of an image-to-3D task (${MESHY_RIG_CREDITS} credits)

Usage:
  shards-asset meshy-rig --task-json meshes/unit.meshy.json --output rigged/unit.glb [--height 1.7]
  shards-asset meshy-rig --task-id <image-to-3d task id> --output rigged/unit.glb

Options:
  --task-json <file>   the .meshy.json record written by "shards-asset mesh"
  --task-id <id>       or the image-to-3D task id directly
  --height <m>         character height in metres, default 1.7
  --output <glb>       default ./output/rigged.glb (an .fbx is saved next to it when Meshy returns one)

If Meshy answers 422 "Pose estimation failed" (a crab-folk with a fused head,
a claw for an arm), the request is not charged: rig the static mesh in Blender
with "shards-asset rig-humanoid" instead.

After a Meshy rig: check the weapon and hand weights (weight-review), then
fix them (rig-fixup: weapon 100% to the hand bone), and repair head weights
(repair-head-weights) before retargeting.`;

export async function runMeshyRig(args) {
  const { values } = parseArgs({
    args,
    options: {
      'task-json': { type: 'string' },
      'task-id': { type: 'string' },
      height: { type: 'string', default: '1.7' },
      output: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help || (!values['task-json'] && !values['task-id'])) {
    console.log(HELP);
    if (!values.help) process.exit(1);
    return;
  }
  const apiKey = process.env.MESHY_API_KEY;
  if (!apiKey) {
    console.error('Error: MESHY_API_KEY environment variable is missing.');
    process.exit(1);
  }
  let taskId = values['task-id'];
  if (!taskId) {
    const rec = JSON.parse(fs.readFileSync(values['task-json'], 'utf-8'));
    taskId = rec.taskId;
    if (!taskId) {
      console.error(`Error: no taskId in ${values['task-json']}`);
      process.exit(1);
    }
  }
  const height = Number(values.height) || 1.7;
  const outputPath = path.resolve(process.cwd(), values.output || './output/rigged.glb');

  console.log(`[Meshy-Rig] Rigging task ${taskId} (height ${height} m, ${MESHY_RIG_CREDITS} credits)`);
  let r;
  try {
    r = await meshyRequest('POST', `${API_ROOT}/rigging`, apiKey, { input_task_id: taskId, height_meters: height });
  } catch (err) {
    console.error(`[Meshy-Rig] Rig request refused: ${err.message}`);
    console.error('[Meshy-Rig] A 4xx refusal is not charged. For a non-standard humanoid use "shards-asset rig-humanoid".');
    process.exit(1);
  }
  const rigTaskId = r.result;
  console.log(`[Meshy-Rig] Rig task ${rigTaskId} created, polling`);
  const t = await pollTask(`${API_ROOT}/rigging/${rigTaskId}`, apiKey, 15_000, 240);
  const res = t.result || {};
  if (!res.rigged_character_glb_url) {
    console.error(`[Meshy-Rig] No rigged_character_glb_url in the result (keys: ${Object.keys(res).join(', ')})`);
    process.exit(1);
  }
  const bytes = await download(res.rigged_character_glb_url, outputPath);
  if (res.rigged_character_fbx_url) {
    try {
      await download(res.rigged_character_fbx_url, outputPath.replace(/\.glb$/i, '.fbx'));
    } catch {
      // the GLB is the source of record
    }
  }
  fs.writeFileSync(outputPath.replace(/\.glb$/i, '') + '.meshy-rig.json', JSON.stringify({
    imageTo3dTaskId: taskId, rigTaskId, heightMeters: height, consumedCredits: t.consumed_credits ?? null,
    finishedAt: new Date().toISOString(),
  }, null, 2));
  console.log(`[Meshy-Rig] Saved ${(bytes / 1048576).toFixed(1)} MB to ${outputPath}`);
}
