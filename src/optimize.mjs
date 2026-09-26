// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Web RTS optimization module.
 * Shrinks textures to 1024px, applies KTX2 Basis Universal compression, and applies meshopt.
 */

import { parseArgs } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

export async function runOptimize(args) {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      'texture-max': { type: 'string', default: '1024' },
      codec: { type: 'string', default: 'mixed' },
      help: { type: 'boolean', short: 'h' },
    }
  });

  if (values.help || !values.input) {
    console.log(`shards-asset optimize: resize textures, transcode to KTX2 and compress geometry with meshopt

Usage:
  shards-asset optimize --input animated.glb --output optimized.glb [--texture-max 1024]

Run it last: after the Blender passes and after dye-mask (dye-mask needs the
uncompressed base colour). Falls back to meshopt only if KTX2 encoding fails.`);
    if (!values.help) process.exit(1);
    return;
  }

  const inputPath = path.resolve(process.cwd(), values.input);
  const outputPath = path.resolve(process.cwd(), values.output || './output/optimized.glb');
  const outDir = path.dirname(outputPath);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const maxTex = parseInt(values['texture-max'], 10) || 1024;

  console.log(`[Optimize] Running web RTS optimization pipeline on: ${inputPath}...`);
  console.log(`[Optimize] Target texture cap: ${maxTex}px | Codec: ${values.codec}`);

  // Test for gltf-transform CLI
  const gtfTest = spawnSync('gltf-transform', ['--version'], { encoding: 'utf-8' });
  const hasGlobalGtf = gtfTest.status === 0;

  // Temporary staging file for the multi-pass pipeline
  const tempResized = outputPath.replace('.glb', '.resized.tmp.glb');

  try {
    const gtfCmd = hasGlobalGtf ? 'gltf-transform' : 'npx @gltf-transform/cli';

    // Step 1: Resize textures
    console.log(`[Optimize] Step 1/3: Resizing textures to max ${maxTex}x${maxTex}...`);
    const resizeArgs = [
      'resize',
      '--width', String(maxTex),
      '--height', String(maxTex),
      inputPath,
      tempResized
    ];
    
    let res = spawnSync(hasGlobalGtf ? 'gltf-transform' : 'npx', hasGlobalGtf ? resizeArgs : ['@gltf-transform/cli', ...resizeArgs], {
      stdio: 'inherit',
      encoding: 'utf-8'
    });

    if (res.status !== 0) {
      console.warn(`[Optimize] gltf-transform resize skipped or failed. Falling back to copy for step 2.`);
      fs.copyFileSync(inputPath, tempResized);
    }

    // Step 2 & 3: Meshopt compression + KTX2 (or gltf-transform optimize)
    console.log(`[Optimize] Step 2/3: Applying meshopt geometry compression and KTX2 GPU textures...`);
    const optArgs = [
      'optimize',
      '--compress', 'meshopt',
      '--texture-compress', 'ktx2',
      // never decimate a rigged unit here: simplify can merge vertices across
      // weight boundaries; decimate on purpose, before rigging, if you need to
      '--simplify', 'false',
      tempResized,
      outputPath
    ];

    res = spawnSync(hasGlobalGtf ? 'gltf-transform' : 'npx', hasGlobalGtf ? optArgs : ['@gltf-transform/cli', ...optArgs], {
      stdio: 'inherit',
      encoding: 'utf-8'
    });

    if (res.status !== 0) {
      console.warn(`[Optimize] KTX2 encoding failed (it needs the KTX-Software "ktx" CLI on your PATH). Applying meshopt only.`);
      const fallbackArgs = ['meshopt', tempResized, outputPath];
      res = spawnSync(hasGlobalGtf ? 'gltf-transform' : 'npx', hasGlobalGtf ? fallbackArgs : ['@gltf-transform/cli', ...fallbackArgs], {
        stdio: 'inherit'
      });
      if (res.status !== 0) throw new Error('gltf-transform meshopt failed');
    }

    // Cleanup temp file
    if (fs.existsSync(tempResized)) {
      fs.unlinkSync(tempResized);
    }

    const inSize = fs.statSync(inputPath).size;
    const outSize = fs.statSync(outputPath).size;
    const savings = (((inSize - outSize) / inSize) * 100).toFixed(1);

    console.log(`[Optimize] Optimization complete!`);
    console.log(`[Optimize] Before: ${(inSize / 1024 / 1024).toFixed(2)} MB -> After: ${(outSize / 1024 / 1024).toFixed(2)} MB (${savings}% smaller)`);
    console.log(`[Optimize] Output saved to: ${outputPath}`);

  } catch (err) {
    console.error(`[Optimize] Error: ${err.message}`);
    if (fs.existsSync(tempResized)) fs.unlinkSync(tempResized);
    process.exit(1);
  }
}
