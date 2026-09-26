// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * dye-mask: positive control on a fixture with KNOWN masked areas.
 *   npm test
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  applyDyeMask, checkDyeMaskJson, createIO, readGlbJson, DYE_MASK_TEXTURE_NAME,
} from '../src/dye_mask.mjs';
import { dyeHsv, decodeUnitDyeMask } from '../src/dye_ramp.mjs';
import { FIXTURE, buildFixtureDocument, regionOf } from './dye_fixture.mjs';

const SIZE = 256;
const ACCENT = 0x2050d0; // blue: far from every fixture hue
const EXTENDED = 0x20a040; // green

async function decodeRgba(bytes) {
  const { data, info } = await sharp(Buffer.from(bytes)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data: new Uint8Array(data), w: info.width, h: info.height };
}

function hueDeg(r, g, b) {
  const [h, s, v] = dyeHsv(r / 255, g / 255, b / 255);
  return { h: h * 360, s, v };
}

test('Meshy key colours round-trip into a one-channel two-tier mask, repainted, on a skinned GLB', async () => {
  const io = await createIO();
  const doc = await buildFixtureDocument({ textureSize: SIZE });
  const inputGlb = await io.writeBinary(doc);
  assert.equal(checkDyeMaskJson(readGlbJson(inputGlb)).ok, false, 'unmasked fixture must fail the check');

  const result = await applyDyeMask(doc, { accentColour: ACCENT, extendedColour: EXTENDED, maskScale: 0.5 });
  const json = readGlbJson(await io.writeBinary(doc));
  const check = checkDyeMaskJson(json);
  assert.ok(check.ok, check.errors.join('; '));
  assert.ok(json.images?.some((i) => i.name === DYE_MASK_TEXTURE_NAME), 'mask image is named');
  assert.equal(json.skins?.length, 1, 'skin preserved');
  assert.equal(json.animations?.length, 1, 'clip preserved');
  assert.ok('JOINTS_0' in json.meshes[0].primitives[0].attributes, 'JOINTS_0 preserved');

  const total = SIZE * SIZE;
  assert.ok(result.accentTexels > total * 0.25 && result.accentTexels < total * 0.31, `accent texels ${result.accentTexels}`);
  assert.ok(result.extendedTexels > total * 0.27 && result.extendedTexels < total * 0.31, `extended texels ${result.extendedTexels}`);

  const mask = await decodeRgba(result.mask.png);
  assert.equal(mask.w, SIZE / 2);
  let accentCore = 0, extCore = 0, redLeak = 0, restLeak = 0, grey = 0;
  for (let y = 2; y < mask.h - 2; y++) {
    for (let x = 0; x < mask.w; x++) {
      const o = (y * mask.w + x) * 4;
      if (mask.data[o] === mask.data[o + 1] && mask.data[o] === mask.data[o + 2]) grey++;
      const u = (x + 0.5) / mask.w;
      const near = [FIXTURE.accent[1], FIXTURE.red[1], FIXTURE.extended[1]].some((b) => Math.abs(u - b) < 3 / mask.w);
      if (near) continue;
      const [ra, ga] = decodeUnitDyeMask(mask.data[o] / 255);
      const r = Math.round(ra * 255), g = Math.round(ga * 255);
      const region = regionOf(u);
      if (region === 'accent') { if (r > 200 && g < 30) accentCore++; else assert.fail(`accent core texel (${x},${y}) mask ${r},${g}`); }
      if (region === 'extended') { if (g > 200 && r < 30) extCore++; else assert.fail(`extended core texel (${x},${y}) mask ${r},${g}`); }
      if (region === 'red' && (r > 20 || g > 20)) redLeak++;
      if (region === 'rest' && (r > 20 || g > 20)) restLeak++;
    }
  }
  assert.ok(accentCore > 0 && extCore > 0, 'positive control found masked texels');
  assert.equal(grey, mask.w * (mask.h - 4), 'mask is one channel (R = G = B)');
  assert.equal(redLeak, 0, 'red cloth next to the accent zone did not key');
  assert.equal(restLeak, 0, 'isolated magenta specks were cleaned up');

  const base = await decodeRgba(doc.getRoot().listTextures().find((t) => t.getName() !== DYE_MASK_TEXTURE_NAME).getImage());
  let fringe = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const o = (y * SIZE + x) * 4;
      const { h, s, v } = hueDeg(base.data[o], base.data[o + 1], base.data[o + 2]);
      if (s > 0.3 && v > 0.15 && Math.abs(h - 300) < 25) fringe++;
      const region = regionOf((x + 0.5) / SIZE);
      const edge = Math.abs((x + 0.5) / SIZE - FIXTURE.accent[1]) < 5 / SIZE;
      if (region === 'accent' && !edge && v > 0.2) assert.ok(Math.abs(h - 220) < 25, `accent repaint hue ${h.toFixed(0)}`);
      if (region === 'red' && !edge) {
        for (let c = 0; c < 3; c++) assert.equal(base.data[o + c], result.review.before[o + c], 'red band untouched');
      }
    }
  }
  assert.equal(fringe, 0, 'no magenta fringe survives the repaint');
});

test('a texture with no key colour is an error, not an empty mask', async () => {
  const doc = await buildFixtureDocument({ textureSize: 64 });
  await assert.rejects(
    applyDyeMask(doc, { rules: [{ tier: 'accent', key: { hue: 60, tol: 5, minSat: 0.99 } }] }),
    /no texel matched/,
  );
});

test('box rules split one colour key into two tiers by model-space region', async () => {
  const doc = await buildFixtureDocument({ textureSize: 128 });
  const res = await applyDyeMask(doc, {
    rules: [
      { tier: 'accent', key: { hue: 0, tol: 20, minSat: 0.5 }, box: { center: [0, 1.5, 0], half: [1, 0.5, 1] }, repaint: false },
      { tier: 'extended', key: { hue: 0, tol: 20, minSat: 0.5 }, box: { center: [0, 0.5, 0], half: [1, 0.5, 1] }, repaint: false },
    ],
  });
  assert.ok(res.accentTexels > 0 && res.extendedTexels > 0);
  const ratio = res.accentTexels / res.extendedTexels;
  assert.ok(ratio > 0.8 && ratio < 1.25, `halves are balanced (${ratio.toFixed(2)})`);
});
