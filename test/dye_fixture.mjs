// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * A small SKINNED, ANIMATED GLB with a base-colour texture laid out in known
 * regions: the positive control for the dye-mask tool. Nothing here is a real unit.
 *
 * Texture (W x W) by column band of the UV square:
 *   [0, 0.30)    ACCENT   magenta #FF00FF, value shaded down the rows, with a
 *                         3-texel Meshy-style blend into the red neighbour
 *   [0.30, 0.40) RED      plain red cloth: must NOT key
 *   [0.40, 0.70) EXTENDED cyan #00FFFF, value shaded
 *   [0.70, 1.0]  REST     gold / steel rows, plus isolated magenta specks that must be cleaned up
 */
import { Document } from '@gltf-transform/core';
import sharp from 'sharp';

export const FIXTURE = {
  accent: [0, 0.3],
  red: [0.3, 0.4],
  extended: [0.4, 0.7],
  rest: [0.7, 1],
  specks: [[0.8, 0.2], [0.9, 0.55], [0.75, 0.8]],
};

function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

export function fixtureTexture(size = 256) {
  const px = new Uint8Array(size * size * 4);
  const red = [0.69, 0.125, 0.125];
  for (let y = 0; y < size; y++) {
    const shade = 0.35 + 0.65 * (y / (size - 1));
    for (let x = 0; x < size; x++) {
      const u = (x + 0.5) / size;
      let c;
      if (u < FIXTURE.accent[1]) {
        c = [shade, 0, shade];
        const edgeX = Math.floor(FIXTURE.accent[1] * size);
        const k = x - (edgeX - 3);
        if (k >= 0) {
          const t = (k + 1) / 4;
          c = [c[0] * (1 - t) + red[0] * shade * t, c[1] * (1 - t) + red[1] * shade * t, c[2] * (1 - t) + red[2] * shade * t];
        }
      } else if (u < FIXTURE.red[1]) {
        c = [red[0] * shade, red[1] * shade, red[2] * shade];
      } else if (u < FIXTURE.extended[1]) {
        c = [0, shade, shade];
      } else {
        c = (y >> 3) % 2 === 0 ? hsvToRgb(0.12, 0.75, 0.55 + 0.4 * shade) : [0.55 * shade + 0.2, 0.57 * shade + 0.2, 0.6 * shade + 0.2];
      }
      const o = (y * size + x) * 4;
      px[o] = Math.round(c[0] * 255);
      px[o + 1] = Math.round(c[1] * 255);
      px[o + 2] = Math.round(c[2] * 255);
      px[o + 3] = 255;
    }
  }
  for (const [fx, fy] of FIXTURE.specks) {
    const o = (Math.floor(fy * size) * size + Math.floor(fx * size)) * 4;
    px[o] = 255; px[o + 1] = 0; px[o + 2] = 255;
  }
  return px;
}

export function regionOf(u) {
  if (u < FIXTURE.accent[1]) return 'accent';
  if (u < FIXTURE.red[1]) return 'red';
  if (u < FIXTURE.extended[1]) return 'extended';
  return 'rest';
}

export async function buildFixtureDocument({ grid = 48, textureSize = 256 } = {}) {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const n = grid + 1;
  const pos = new Float32Array(n * n * 3);
  const nrm = new Float32Array(n * n * 3);
  const uv = new Float32Array(n * n * 2);
  const joints = new Uint16Array(n * n * 4);
  const weights = new Float32Array(n * n * 4);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i;
      const x = i / grid, y = j / grid;
      pos.set([x - 0.5, y * 2, 0], k * 3);
      nrm.set([0, 0, 1], k * 3);
      uv.set([x, 1 - y], k * 2);
      joints.set([0, 1, 0, 0], k * 4);
      weights.set([1 - y, y, 0, 0], k * 4);
    }
  }
  const idx = new Uint32Array(grid * grid * 6);
  let t = 0;
  for (let j = 0; j < grid; j++) {
    for (let i = 0; i < grid; i++) {
      const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
      idx.set([a, b, d, a, d, c], t);
      t += 6;
    }
  }
  const acc = (arr, type) => doc.createAccessor().setArray(arr).setType(type).setBuffer(buffer);
  const png = await sharp(Buffer.from(fixtureTexture(textureSize)), { raw: { width: textureSize, height: textureSize, channels: 4 } }).png().toBuffer();
  const tex = doc.createTexture('texture_0').setImage(new Uint8Array(png)).setMimeType('image/png');
  const mat = doc.createMaterial('Material_1').setBaseColorTexture(tex).setRoughnessFactor(1).setMetallicFactor(0);
  const prim = doc.createPrimitive()
    .setAttribute('POSITION', acc(pos, 'VEC3'))
    .setAttribute('NORMAL', acc(nrm, 'VEC3'))
    .setAttribute('TEXCOORD_0', acc(uv, 'VEC2'))
    .setAttribute('JOINTS_0', acc(joints, 'VEC4'))
    .setAttribute('WEIGHTS_0', acc(weights, 'VEC4'))
    .setIndices(acc(idx, 'SCALAR'))
    .setMaterial(mat);
  const mesh = doc.createMesh('fixture').addPrimitive(prim);
  const root = doc.createNode('Hips').setTranslation([0, 0, 0]);
  const tip = doc.createNode('Spine').setTranslation([0, 1, 0]);
  root.addChild(tip);
  const ibm = new Float32Array([
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1,
  ]);
  const skin = doc.createSkin('Armature').addJoint(root).addJoint(tip).setSkeleton(root)
    .setInverseBindMatrices(acc(ibm, 'MAT4'));
  const body = doc.createNode('Body').setMesh(mesh).setSkin(skin);
  const armature = doc.createNode('Armature').addChild(root).addChild(body);
  doc.createScene('Scene').addChild(armature);
  const times = acc(new Float32Array([0, 0.5, 1]), 'SCALAR');
  const rots = acc(new Float32Array([0, 0, 0, 1, 0, 0, 0.3826834, 0.9238795, 0, 0, 0, 1]), 'VEC4');
  const sampler = doc.createAnimationSampler().setInput(times).setOutput(rots).setInterpolation('LINEAR');
  const channel = doc.createAnimationChannel().setTargetNode(tip).setTargetPath('rotation').setSampler(sampler);
  doc.createAnimation('idle').addSampler(sampler).addChannel(channel);
  return doc;
}
