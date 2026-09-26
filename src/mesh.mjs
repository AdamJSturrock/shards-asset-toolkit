/**
 * Meshy API integration module for 2D-to-3D mesh generation.
 * Handles task submission, progress polling, and GLB download.
 */

import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';

const MESHY_BASE_URL = 'https://api.meshy.ai/v2';

export async function runMesh(args) {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      polycount: { type: 'string', default: '30000' },
      'texture-size': { type: 'string', default: '2048' },
    }
  });

  if (!values.input) {
    console.error('Error: --input <path-to-image.png> is required');
    process.exit(1);
  }

  const apiKey = process.env.MESHY_API_KEY;
  if (!apiKey) {
    console.error('Error: MESHY_API_KEY environment variable is missing.');
    console.error('Please obtain a key from https://www.meshy.ai/ and set it in your .env');
    process.exit(1);
  }

  const inputPath = path.resolve(process.cwd(), values.input);
  if (!fs.existsSync(inputPath)) {
    console.error(`Error: Input file does not exist: ${inputPath}`);
    process.exit(1);
  }

  const outputPath = values.output || './output/mesh.glb';
  const outDir = path.dirname(path.resolve(process.cwd(), outputPath));
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  // Convert input image to data URI
  const imageBuffer = fs.readFileSync(inputPath);
  const ext = path.extname(inputPath).toLowerCase().replace('.', '');
  const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
  const dataUri = `data:${mimeType};base64,${imageBuffer.toString('base64')}`;

  console.log(`[Meshy] Submitting image-to-3d task (${(imageBuffer.length / 1024).toFixed(1)} KB)...`);
  console.log(`[Meshy] Target polycount: ${values.polycount} | Texture size: ${values['texture-size']}`);

  const createPayload = {
    image_url: dataUri,
    target_polycount: parseInt(values.polycount, 10) || 30000,
    enable_pbr: true,
    ai_model: 'latest',
    texture_richness: 'high'
  };

  const createRes = await fetch(`${MESHY_BASE_URL}/image-to-3d`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(createPayload)
  });

  if (!createRes.ok) {
    const errorText = await createRes.text();
    console.error(`[Meshy] Failed to create task (${createRes.status}): ${errorText}`);
    process.exit(1);
  }

  const { result: taskId } = await createRes.json();
  console.log(`[Meshy] Task created successfully. Task ID: ${taskId}`);
  console.log(`[Meshy] Polling for progress (typically takes 60-120 seconds)...`);

  // Poll for status
  let finished = false;
  let attempts = 0;
  let glbUrl = null;

  while (!finished && attempts < 120) {
    attempts++;
    await new Promise((r) => setTimeout(r, 4000));

    const pollRes = await fetch(`${MESHY_BASE_URL}/image-to-3d/${taskId}`, {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    });

    if (!pollRes.ok) {
      console.warn(`[Meshy] Warning: Poll check failed (${pollRes.status}), retrying...`);
      continue;
    }

    const task = await pollRes.json();
    const progress = task.progress || 0;
    const status = task.status;

    process.stdout.write(`\r[Meshy] Status: ${status} (${progress}%)`);

    if (status === 'SUCCEEDED') {
      finished = true;
      glbUrl = task.model_urls?.glb;
      process.stdout.write('\n');
      console.log(`[Meshy] 3D generation succeeded!`);
      break;
    } else if (status === 'FAILED' || status === 'EXPIRED') {
      process.stdout.write('\n');
      console.error(`[Meshy] Task failed: ${task.task_error?.message || 'Unknown error'}`);
      process.exit(1);
    }
  }

  if (!glbUrl) {
    console.error('\n[Meshy] Timed out waiting for task completion.');
    process.exit(1);
  }

  console.log(`[Meshy] Downloading GLB to: ${outputPath}...`);
  const downloadRes = await fetch(glbUrl);
  if (!downloadRes.ok) {
    console.error(`[Meshy] Failed to download GLB file: ${downloadRes.statusText}`);
    process.exit(1);
  }

  const arrayBuffer = await downloadRes.arrayBuffer();
  fs.writeFileSync(outputPath, Buffer.from(arrayBuffer));
  console.log(`[Meshy] Saved ${arrayBuffer.byteLength} bytes to ${outputPath}`);
}
