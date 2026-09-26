// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Meshy image-to-3D: submit a concept image, poll, download the GLB.
 * POST https://api.meshy.ai/openapi/v1/image-to-3d (30 credits per mesh).
 *
 * Writes <output>.glb and <output>.meshy.json (the task id and settings), so
 * `shards-asset meshy-rig --task-json <output>.meshy.json` can auto-rig the
 * same task without re-uploading anything.
 */

import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';

const MESHY_API = 'https://api.meshy.ai/openapi/v1/image-to-3d';
export const MESHY_IMAGE_TO_3D_CREDITS = 30;

const HELP = `shards-asset mesh: convert a concept image into a textured GLB with Meshy (${MESHY_IMAGE_TO_3D_CREDITS} credits)

Usage:
  shards-asset mesh --input concept.png --output meshes/unit.glb [options]

Options:
  --input <png|jpg>        concept image (required)
  --output <glb>           default ./output/mesh.glb
  --polycount <n>          target_polycount, default 30000
  --ai-model <m>           meshy-6-lite | meshy-6 | meshy-7.1 | latest (default: Meshy's default, latest)
  --pose-mode <p>          a-pose | t-pose (default: off). See the warning below.
  --symmetry <m>           off | auto | on (default off). "on" helps a bilateral creature drawn in 3/4
                           view, whose hidden side otherwise comes back with fewer legs.
  --pbr                    ask Meshy for PBR maps (default off: base colour only)

Pose mode and held weapons:
  pose_mode t-pose re-poses the character into Meshy's own T-pose with open, flat
  hands and silently DROPS whatever the character holds. For an armed humanoid,
  draw the concept in a T-pose with the weapon in hand and leave --pose-mode off:
  Meshy's auto-rig still accepts the mesh. Then weight the weapon 100% to the
  hand bone (shards-asset rig-fixup), because Meshy smears weapon weights onto
  the forearm or upper arm.`;

export async function runMesh(args) {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      polycount: { type: 'string', default: '30000' },
      'ai-model': { type: 'string' },
      'pose-mode': { type: 'string' },
      symmetry: { type: 'string', default: 'off' },
      pbr: { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help || !values.input) {
    console.log(HELP);
    if (!values.help) process.exit(1);
    return;
  }
  const poseMode = values['pose-mode'] || '';
  if (!['', 'a-pose', 't-pose'].includes(poseMode)) {
    console.error('Error: --pose-mode must be a-pose or t-pose');
    process.exit(1);
  }
  const aiModel = values['ai-model'] || null;
  if (aiModel && !['meshy-6-lite', 'meshy-6', 'meshy-7.1', 'latest'].includes(aiModel)) {
    console.error('Error: --ai-model must be meshy-6-lite | meshy-6 | meshy-7.1 | latest');
    process.exit(1);
  }
  if (!['off', 'auto', 'on'].includes(values.symmetry)) {
    console.error('Error: --symmetry must be off | auto | on');
    process.exit(1);
  }
  if (poseMode === 't-pose') {
    console.warn('[Meshy] WARNING: pose_mode t-pose drops held weapons and opens the hands. '
      + 'For an armed character use T-pose concept art and leave --pose-mode off.');
  }

  const apiKey = process.env.MESHY_API_KEY;
  if (!apiKey) {
    console.error('Error: MESHY_API_KEY environment variable is missing.');
    console.error('Obtain a key from https://www.meshy.ai/ and set it in your .env');
    process.exit(1);
  }

  const inputPath = path.resolve(process.cwd(), values.input);
  if (!fs.existsSync(inputPath)) {
    console.error(`Error: Input file does not exist: ${inputPath}`);
    process.exit(1);
  }
  const outputPath = path.resolve(process.cwd(), values.output || './output/mesh.glb');
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const imageBuffer = fs.readFileSync(inputPath);
  const ext = path.extname(inputPath).toLowerCase();
  const mimeType = ext === '.png' ? 'image/png' : 'image/jpeg';
  const dataUri = `data:${mimeType};base64,${imageBuffer.toString('base64')}`;

  const body = {
    image_url: dataUri,
    texture_image_url: dataUri,
    enable_pbr: values.pbr,
    target_formats: ['glb'],
    symmetry_mode: values.symmetry,
    should_remesh: true,
    target_polycount: parseInt(values.polycount, 10) || 30000,
  };
  if (poseMode) body.pose_mode = poseMode;
  if (aiModel) body.ai_model = aiModel;

  console.log(`[Meshy] Submitting image-to-3d task (${(imageBuffer.length / 1024).toFixed(1)} KB, ${MESHY_IMAGE_TO_3D_CREDITS} credits)`);
  console.log(`[Meshy] polycount=${body.target_polycount} symmetry=${body.symmetry_mode} pose_mode=${poseMode || '(off)'} ai_model=${aiModel || '(default)'}`);

  const taskId = await meshyRequest('POST', MESHY_API, apiKey, body).then((j) => j.result);
  if (!taskId) {
    console.error('[Meshy] POST succeeded but returned no task id');
    process.exit(1);
  }
  const record = {
    taskId,
    input: path.relative(process.cwd(), inputPath),
    settings: { ...body, image_url: undefined, texture_image_url: undefined },
    creditsRequested: MESHY_IMAGE_TO_3D_CREDITS,
    startedAt: new Date().toISOString(),
  };
  const recordPath = outputPath.replace(/\.glb$/i, '') + '.meshy.json';
  fs.writeFileSync(recordPath, JSON.stringify(record, null, 2));
  console.log(`[Meshy] Task ${taskId} created; record written to ${recordPath}`);

  const task = await pollTask(`${MESHY_API}/${taskId}`, apiKey, 4000, 150);
  const glbUrl = task.model_urls?.glb;
  if (!glbUrl) {
    console.error(`[Meshy] Task finished without a GLB url (status ${task.status})`);
    process.exit(1);
  }
  const bytes = await download(glbUrl, outputPath);
  record.status = task.status;
  record.finishedAt = new Date().toISOString();
  record.consumedCredits = task.consumed_credits ?? null;
  fs.writeFileSync(recordPath, JSON.stringify(record, null, 2));
  console.log(`[Meshy] Saved ${(bytes / 1048576).toFixed(1)} MB to ${outputPath}`);
}

// ─── shared Meshy helpers (also used by meshy-rig) ─────────────────────────

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function meshyRequest(method, url, apiKey, body) {
  for (let attempt = 0; attempt < 30; attempt++) {
    const res = await fetch(url, {
      method,
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.ok) return res.json();
    const text = await res.text();
    if (res.status === 402) throw new Error(`Meshy 402 (insufficient credits): ${text.slice(0, 200)}`);
    if (res.status === 429) {
      const wait = text.includes('NoMoreConcurrentTasks') ? 60_000 : 3_000;
      console.log(`[Meshy] 429 ${text.slice(0, 60)}: waiting ${wait / 1000}s`);
      await sleep(wait);
      continue;
    }
    if (res.status >= 500) {
      await sleep(5_000);
      continue;
    }
    throw new Error(`Meshy ${method} ${url.replace(/^https:\/\/api\.meshy\.ai/, '')} -> ${res.status}: ${text.slice(0, 300)}`);
  }
  throw new Error('Meshy request: retries exhausted');
}

export async function pollTask(url, apiKey, intervalMs, maxPolls) {
  let last = -1;
  for (let i = 0; i < maxPolls; i++) {
    const t = await meshyRequest('GET', url, apiKey);
    if (t.status === 'SUCCEEDED') {
      process.stdout.write('\n');
      return t;
    }
    if (['FAILED', 'CANCELED', 'EXPIRED'].includes(t.status)) {
      process.stdout.write('\n');
      throw new Error(`Meshy task ${t.status}: ${JSON.stringify(t.task_error || {})}`);
    }
    if (t.progress !== last) {
      process.stdout.write(`\r[Meshy] ${t.status} ${t.progress ?? 0}%   `);
      last = t.progress;
    }
    await sleep(intervalMs);
  }
  throw new Error('Meshy task: timed out while polling');
}

export async function download(url, dest) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (${res.status})`);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, buf);
  return buf.length;
}
