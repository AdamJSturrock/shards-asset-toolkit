// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * dye-mask: extract a unit's player-colour DYE MASK from its base-colour
 * texture and write it into the GLB in the Shards of Stone runtime format:
 *
 *   occlusionTexture = `sos_dye_mask` PNG, ONE channel: 0 none, 0.5 extended, 1 accent
 *   material.extras.sosDye = { v: 1, lum: { accent: [lo, hi], extended: [lo, hi] } }
 *
 * One channel because KTX2 stores an occlusion texture as R-only: the game's
 * texture pipeline would drop a mask spread over G or B. The mask is written
 * grey (R = G = B) so any R-only encode keeps all of it.
 *
 * Meshy convention: in the concept image you feed Meshy, paint accent zones
 * (banners, plumes, sashes) pure MAGENTA #FF00FF and extended zones (a cloak,
 * a caparison) pure CYAN #00FFFF. Those texels are keyed softly (Meshy blends
 * edges), un-mixed, and REPAINTED in a default owner colour, so the unit still
 * looks finished where no player colour applies.
 *
 * Pixel work only, in UV space: skeleton, skin, clips and geometry pass
 * through gltf-transform untouched, so run it on the final rigged, animated
 * GLB AFTER the Blender passes (a Blender re-export afterwards drops the
 * extras: run `verify` after any later step). Run it before KTX2 compression.
 *
 * Rules (for existing art without key colours), a JSON array of:
 *   { tier: 'accent' | 'extended',
 *     key: 'magenta' | 'cyan' | { hue: deg, tol: deg, minSat, minValue? } | { luma: [lo, hi], maxSat, minSat? },
 *     box?: { center: [x, y, z], half: [x, y, z] },   // mesh-local bind pose, glTF Y-up (see `bbox`)
 *     repaint?: boolean }                            // default true for magenta/cyan keys
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import {
  dyeBigAreaColor, dyeHsv, dyeRampBigArea, dyeRampCloth, dyeShade, hexToRgb01, encodeUnitDyeMask,
} from './dye_ramp.mjs';

export const DYE_MASK_TEXTURE_NAME = 'sos_dye_mask';
export const DYE_EXTRAS_KEY = 'sosDye';
export const MESHY_KEY_RULES = [
  { tier: 'accent', key: 'magenta' },
  { tier: 'extended', key: 'cyan' },
];

let _sharp;
async function sharpLib() {
  if (!_sharp) {
    try {
      _sharp = (await import('sharp')).default;
    } catch {
      throw new Error('dye-mask needs the "sharp" package: run npm install');
    }
  }
  return _sharp;
}

export async function createIO() {
  const { NodeIO } = await import('@gltf-transform/core');
  const { ALL_EXTENSIONS } = await import('@gltf-transform/extensions');
  const { MeshoptDecoder, MeshoptEncoder } = await import('meshoptimizer');
  await MeshoptDecoder.ready;
  await MeshoptEncoder.ready;
  return new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });
}

// ─── Keys ───────────────────────────────────────────────────────────────────

function smoothstep(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function hueDistanceDeg(a, b) {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

const isKeyColour = (key) => key === 'magenta' || key === 'cyan';
const keyRgb = (key) => (key === 'magenta' ? [1, 0, 1] : [0, 1, 1]);

/** Soft key weight of one sRGB texel (0..1 channels). */
export function keyWeight(key, r, g, b) {
  const [h, s, v] = dyeHsv(r, g, b);
  if (isKeyColour(key)) {
    // Tolerant of Meshy shading (value varies) and edge blending (hue drifts,
    // saturation drops): partial weights, un-mixed on repaint.
    const target = key === 'magenta' ? 300 : 180;
    const hw = 1 - smoothstep(22, 45, hueDistanceDeg(h * 360, target));
    return hw * smoothstep(0.22, 0.5, s) * smoothstep(0.06, 0.14, v);
  }
  if ('hue' in key) {
    const minValue = key.minValue ?? 0.06;
    const hw = 1 - smoothstep(key.tol * 0.7, key.tol, hueDistanceDeg(h * 360, key.hue));
    return hw * smoothstep(key.minSat * 0.8, key.minSat, s) * smoothstep(minValue * 0.7, minValue, v);
  }
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return smoothstep(key.luma[0] - 0.03, key.luma[0], luma)
    * (1 - smoothstep(key.luma[1], key.luma[1] + 0.03, luma))
    * (1 - smoothstep(key.maxSat, key.maxSat + 0.05, s))
    * (key.minSat ? smoothstep(key.minSat * 0.7, key.minSat, s) : 1);
}

// ─── UV -> model-space position (for box rules) ─────────────────────────────

/** Rasterise the primitives using `material` into UV space: bind-pose position per texel (NaN = uncovered). */
export function rasterizeUvPositions(doc, material, w, h) {
  const pos = new Float32Array(w * h * 3).fill(Number.NaN);
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      if (prim.getMaterial() !== material) continue;
      const P = prim.getAttribute('POSITION');
      const T = prim.getAttribute('TEXCOORD_0');
      if (!P || !T) continue;
      const idx = prim.getIndices();
      const count = idx ? idx.getCount() : P.getCount();
      const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
      const ua = [0, 0], ub = [0, 0], uc = [0, 0];
      for (let t = 0; t + 2 < count; t += 3) {
        const i0 = idx ? idx.getScalar(t) : t;
        const i1 = idx ? idx.getScalar(t + 1) : t + 1;
        const i2 = idx ? idx.getScalar(t + 2) : t + 2;
        P.getElement(i0, a); P.getElement(i1, b); P.getElement(i2, c);
        T.getElement(i0, ua); T.getElement(i1, ub); T.getElement(i2, uc);
        const x0 = ua[0] * w, y0 = ua[1] * h;
        const x1 = ub[0] * w, y1 = ub[1] * h;
        const x2 = uc[0] * w, y2 = uc[1] * h;
        const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (Math.abs(area) < 1e-9) continue;
        const minX = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
        const maxX = Math.min(w - 1, Math.ceil(Math.max(x0, x1, x2)));
        const minY = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
        const maxY = Math.min(h - 1, Math.ceil(Math.max(y0, y1, y2)));
        const slack = 0.5 / Math.max(1, Math.sqrt(Math.abs(area)));
        for (let py = minY; py <= maxY; py++) {
          for (let px = minX; px <= maxX; px++) {
            const sx = px + 0.5, sy = py + 0.5;
            let w0 = ((x1 - sx) * (y2 - sy) - (x2 - sx) * (y1 - sy)) / area;
            let w1 = ((x2 - sx) * (y0 - sy) - (x0 - sx) * (y2 - sy)) / area;
            let w2 = 1 - w0 - w1;
            if (w0 < -slack || w1 < -slack || w2 < -slack) continue;
            w0 = Math.max(0, w0); w1 = Math.max(0, w1); w2 = Math.max(0, w2);
            const sum = w0 + w1 + w2 || 1;
            const o = (py * w + px) * 3;
            pos[o] = (a[0] * w0 + b[0] * w1 + c[0] * w2) / sum;
            pos[o + 1] = (a[1] * w0 + b[1] * w1 + c[1] * w2) / sum;
            pos[o + 2] = (a[2] * w0 + b[2] * w1 + c[2] * w2) / sum;
          }
        }
      }
    }
  }
  return pos;
}

// ─── Mask extraction ────────────────────────────────────────────────────────

export function computeTierWeights(rgba, w, h, rules, positions) {
  const n = w * h;
  const accent = new Float32Array(n);
  const extended = new Float32Array(n);
  const rule = new Int16Array(n).fill(-1);
  const best = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] / 255, g = rgba[i * 4 + 1] / 255, b = rgba[i * 4 + 2] / 255;
    for (let k = 0; k < rules.length; k++) {
      const ru = rules[k];
      if (ru.box) {
        if (!positions) continue;
        const px = positions[i * 3];
        if (Number.isNaN(px)) continue;
        if (Math.abs(px - ru.box.center[0]) > ru.box.half[0]
          || Math.abs(positions[i * 3 + 1] - ru.box.center[1]) > ru.box.half[1]
          || Math.abs(positions[i * 3 + 2] - ru.box.center[2]) > ru.box.half[2]) continue;
      }
      const kw = keyWeight(ru.key, r, g, b);
      if (kw > best[i]) {
        best[i] = kw;
        rule[i] = k;
      }
    }
  }
  const rawRule = new Int16Array(rule);
  // Speck cleanup: a texel keeps its weight only if at least 3 of its 8
  // neighbours are keyed too (isolated noise texels in Meshy bakes).
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (best[i] <= 0) continue;
      let nb = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          if (best[yy * w + xx] > 0.25) nb++;
        }
      }
      const wgt = nb >= 3 ? best[i] : 0;
      const ru = rule[i] >= 0 ? rules[rule[i]] : null;
      if (!ru || wgt <= 0) { rule[i] = -1; continue; }
      if (ru.tier === 'accent') accent[i] = wgt;
      else extended[i] = wgt;
    }
  }
  return { accent, extended, rule, rawWeight: best, rawRule };
}

/** [p5, p95] of HSV value over texels whose weight > 0.5. */
export function measureLum(rgba, weights) {
  const hist = new Uint32Array(256);
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] <= 0.5) continue;
    hist[Math.max(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2])]++;
    total++;
  }
  if (!total) return undefined;
  let acc = 0, lo = 0, hi = 255, gotLo = false;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (!gotLo && acc >= total * 0.05) { lo = v; gotLo = true; }
    if (acc >= total * 0.95) { hi = v; break; }
  }
  return [lo / 255, Math.max(hi, lo + 1) / 255];
}

/** Repaint keyed texels in the tier's default colour, un-mixing the key colour: out = orig + w * (ramp - key * value). */
export function repaintKeyed(rgba, weights, rules, accentColour, extendedColour) {
  const accLum = measureLum(rgba, weights.accent) ?? [0.2, 0.9];
  const extLum = measureLum(rgba, weights.extended) ?? [0.2, 0.9];
  const accC = hexToRgb01(accentColour);
  const extC = dyeBigAreaColor(hexToRgb01(extendedColour));
  const out = [0, 0, 0];
  for (let i = 0; i < weights.rawRule.length; i++) {
    const k = weights.rawRule[i];
    if (k < 0) continue;
    const ru = rules[k];
    if (!(ru.repaint ?? isKeyColour(ru.key))) continue;
    const wgt = weights.rawWeight[i];
    if (wgt <= 0) continue;
    const r = rgba[i * 4] / 255, g = rgba[i * 4 + 1] / 255, b = rgba[i * 4 + 2] / 255;
    const value = Math.max(r, g, b);
    if (ru.tier === 'accent') dyeRampCloth(accC, dyeShade(value, accLum[0], accLum[1]), out);
    else dyeRampBigArea(extC, dyeShade(value, extLum[0], extLum[1]), out);
    const kc = isKeyColour(ru.key) ? keyRgb(ru.key)
      : [r / Math.max(value, 1e-4), g / Math.max(value, 1e-4), b / Math.max(value, 1e-4)];
    const src = [r, g, b];
    for (let c = 0; c < 3; c++) {
      const v = src[c] + wgt * (out[c] - kc[c] * value);
      rgba[i * 4 + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
  }
}

// ─── GLB driver ─────────────────────────────────────────────────────────────

async function decode(tex) {
  const bytes = tex.getImage();
  if (!bytes) throw new Error(`texture "${tex.getName()}" has no image`);
  if (tex.getMimeType() === 'image/ktx2') {
    throw new Error('base colour is KTX2: run dye-mask on the uncompressed GLB, before "optimize"');
  }
  const sharp = await sharpLib();
  const { data, info } = await sharp(Buffer.from(bytes)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), w: info.width, h: info.height };
}

function overlayPanel(before, weights) {
  const out = new Uint8Array(before.length);
  for (let i = 0; i < weights.rule.length; i++) {
    const a = weights.accent[i], e = weights.extended[i];
    for (let c = 0; c < 3; c++) {
      const dim = before[i * 4 + c] * 0.35;
      const mag = c === 1 ? 0 : 255;
      const cyn = c === 0 ? 0 : 255;
      out[i * 4 + c] = Math.round(dim * (1 - a - e) + mag * a + cyn * e);
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * Add a dye mask to every base-coloured material of `doc` (in place).
 * Throws if no texel matched any rule: a mask of nothing is a broken input.
 */
export async function applyDyeMask(doc, opts = {}) {
  const sharp = await sharpLib();
  const rules = opts.rules ?? MESHY_KEY_RULES;
  const maskScale = opts.maskScale ?? 0.5;
  const accentColour = opts.accentColour ?? 0xb3202a;
  const extendedColour = opts.extendedColour ?? opts.accentColour ?? 0xb3202a;
  const needPositions = rules.some((r) => r.box);

  const byTexture = new Map();
  for (const m of doc.getRoot().listMaterials()) {
    const t = m.getBaseColorTexture();
    if (!t) continue;
    const list = byTexture.get(t) ?? [];
    list.push(m);
    byTexture.set(t, list);
  }
  if (byTexture.size === 0) throw new Error('no material with a base colour texture');
  if (byTexture.size > 1) throw new Error(`${byTexture.size} base colour textures: one per unit is supported (Meshy units have one)`);

  const [[tex, materials]] = [...byTexture.entries()];
  const { data, w, h } = await decode(tex);
  const before = new Uint8Array(data);
  let positions = null;
  if (needPositions) {
    positions = new Float32Array(w * h * 3).fill(Number.NaN);
    for (const m of materials) {
      const p = rasterizeUvPositions(doc, m, w, h);
      for (let i = 0; i < p.length; i += 3) {
        if (!Number.isNaN(p[i])) { positions[i] = p[i]; positions[i + 1] = p[i + 1]; positions[i + 2] = p[i + 2]; }
      }
    }
  }
  const weights = computeTierWeights(data, w, h, rules, positions);
  let accentTexels = 0, extendedTexels = 0;
  for (let i = 0; i < weights.rule.length; i++) {
    if (weights.accent[i] > 0.5) accentTexels++;
    if (weights.extended[i] > 0.5) extendedTexels++;
  }
  if (accentTexels + extendedTexels === 0) throw new Error('no texel matched any dye rule: nothing to mask');

  repaintKeyed(data, weights, rules, accentColour, extendedColour);
  const lum = { accent: measureLum(data, weights.accent), extended: measureLum(data, weights.extended) };

  const maskFull = Buffer.alloc(w * h * 4);
  for (let i = 0; i < weights.rule.length; i++) {
    const v = Math.round(encodeUnitDyeMask(weights.accent[i], weights.extended[i]) * 255);
    maskFull[i * 4] = v;
    maskFull[i * 4 + 1] = v;
    maskFull[i * 4 + 2] = v;
    maskFull[i * 4 + 3] = 255;
  }
  const mw = Math.max(1, Math.round(w * maskScale));
  const mh = Math.max(1, Math.round(h * maskScale));
  const maskPng = await sharp(maskFull, { raw: { width: w, height: h, channels: 4 } })
    .resize(mw, mh, { kernel: 'lanczos2' })
    .removeAlpha()
    .png({ compressionLevel: 9 })
    .toBuffer();

  const mime = tex.getMimeType();
  const enc = sharp(Buffer.from(data), { raw: { width: w, height: h, channels: 4 } });
  const baseBytes = mime === 'image/jpeg' ? await enc.removeAlpha().jpeg({ quality: 95 }).toBuffer() : await enc.png().toBuffer();
  tex.setImage(new Uint8Array(baseBytes));

  const maskTex = doc.createTexture(DYE_MASK_TEXTURE_NAME)
    .setImage(new Uint8Array(maskPng))
    .setMimeType('image/png')
    .setURI(`${DYE_MASK_TEXTURE_NAME}.png`);
  const lumOut = {};
  if (lum.accent) lumOut.accent = lum.accent;
  if (lum.extended) lumOut.extended = lum.extended;
  for (const m of materials) {
    m.setOcclusionTexture(maskTex);
    m.getOcclusionTextureInfo()?.setTexCoord(0);
    m.setOcclusionStrength(1);
    m.setExtras({ ...(m.getExtras() ?? {}), [DYE_EXTRAS_KEY]: { v: 1, lum: lumOut } });
  }

  return {
    materials: materials.map((m) => m.getName()),
    accentTexels,
    extendedTexels,
    lum: lumOut,
    review: { width: w, height: h, before, overlay: overlayPanel(before, weights), after: new Uint8Array(data) },
    mask: { width: mw, height: mh, png: new Uint8Array(maskPng) },
  };
}

// ─── Verification ───────────────────────────────────────────────────────────

/** Does this GLB JSON carry a runtime-valid dye mask? Works on PNG, KTX2 or an external uri. */
export function checkDyeMaskJson(json) {
  const errors = [];
  const mats = json.materials ?? [];
  const dyed = mats.filter((m) => m.extras && DYE_EXTRAS_KEY in m.extras);
  if (dyed.length === 0) errors.push(`no material carries extras.${DYE_EXTRAS_KEY}`);
  const srcOf = (texIndex) => {
    const t = json.textures?.[texIndex];
    return t?.source ?? t?.extensions?.KHR_texture_basisu?.source ?? t?.extensions?.EXT_texture_webp?.source;
  };
  for (const m of dyed) {
    const occ = m.occlusionTexture?.index;
    if (occ === undefined) { errors.push('dye-marked material has no occlusionTexture'); continue; }
    const base = m.pbrMetallicRoughness?.baseColorTexture?.index;
    const occSrc = srcOf(occ);
    if (occSrc === undefined) errors.push('occlusion texture has no image source');
    if (base !== undefined && srcOf(base) === occSrc) errors.push('occlusion slot points at the BASE colour image (mask dropped)');
    const img = occSrc !== undefined ? json.images?.[occSrc] : undefined;
    if (img && img.name !== undefined && img.name !== DYE_MASK_TEXTURE_NAME) {
      errors.push(`occlusion image is named "${img.name}", expected "${DYE_MASK_TEXTURE_NAME}"`);
    }
  }
  return { ok: errors.length === 0, errors };
}

/** Parse a GLB's JSON chunk. */
export function readGlbJson(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  const len = dv.getUint32(12, true);
  return JSON.parse(Buffer.from(bytes.subarray(20, 20 + len)).toString('utf8'));
}

export async function writeReviewPng(result, outPath, panel = 512) {
  const sharp = await sharpLib();
  const { width, height } = result.review;
  const panels = await Promise.all([result.review.before, result.review.overlay, result.review.after].map((px) =>
    sharp(Buffer.from(px), { raw: { width, height, channels: 4 } }).resize(panel, panel, { fit: 'fill' }).png().toBuffer()));
  await sharp({ create: { width: panel * 3 + 16, height: panel, channels: 4, background: { r: 24, g: 24, b: 28, alpha: 1 } } })
    .composite(panels.map((input, i) => ({ input, left: i * (panel + 8), top: 0 })))
    .png()
    .toFile(outPath);
}

// ─── CLI ────────────────────────────────────────────────────────────────────

const HELP = `shards-asset dye-mask: author and verify a unit's player-colour dye mask

Usage:
  # Meshy convention: accent zones painted #FF00FF, extended (big cloth) zones #00FFFF in the concept
  shards-asset dye-mask apply --in character.glb --out character.dyed.glb \\
      [--accent-colour '#b3202a'] [--extended-colour '#7a1a1a'] [--mask-scale 0.5] [--review review.png] [--render3d]

  # Existing art without key colours: rules from a JSON file (see src/dye_mask.mjs)
  shards-asset dye-mask apply --in a.glb --out b.glb --rules rules.json

  # Print the mesh-local bind-pose bounding box (to author box rules)
  shards-asset dye-mask bbox --in a.glb

  # Check GLBs still carry the mask (exit 1 on failure)
  shards-asset dye-mask verify a.glb [b.glb ...]

Output format: the glTF occlusion slot holds ONE channel of levels,
0 = no dye, 0.5 = extended, 1 = accent, because KTX2 stores an occlusion
texture as R-only. material.extras.sosDye carries the measured value ranges.

Run "apply" on the final rigged and animated GLB, after every Blender pass
(a Blender re-export drops the extras) and before "optimize". --render3d also
renders the colour-coded mask on the model from four sides (needs Blender).`;

function arg(args, flag) {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

function parseHex(s) {
  if (!s) return undefined;
  const m = /^#?([0-9a-f]{6})$/i.exec(s.trim());
  if (!m) throw new Error(`bad colour "${s}" (want #rrggbb)`);
  return parseInt(m[1], 16);
}

async function cmdApply(args) {
  const input = arg(args, '--in');
  const output = arg(args, '--out');
  if (!input || !output) throw new Error('apply needs --in and --out');
  const rulesPath = arg(args, '--rules');
  const rules = rulesPath ? JSON.parse(fs.readFileSync(rulesPath, 'utf8')) : MESHY_KEY_RULES;
  const io = await createIO();
  const doc = await io.read(input);
  const result = await applyDyeMask(doc, {
    rules,
    accentColour: parseHex(arg(args, '--accent-colour')),
    extendedColour: parseHex(arg(args, '--extended-colour')),
    maskScale: arg(args, '--mask-scale') ? Number(arg(args, '--mask-scale')) : undefined,
  });
  fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
  await io.write(output, doc);
  const check = checkDyeMaskJson(readGlbJson(new Uint8Array(fs.readFileSync(output))));
  if (!check.ok) throw new Error(`written GLB failed verification: ${check.errors.join('; ')}`);
  console.log(`[Dye] wrote ${output}`);
  console.log(`[Dye] materials ${result.materials.join(', ')}; accent ${result.accentTexels} texels; extended ${result.extendedTexels} texels`);
  console.log(`[Dye] mask ${result.mask.width}x${result.mask.height}; lum ${JSON.stringify(result.lum)}`);
  const review = arg(args, '--review');
  if (review) {
    await writeReviewPng(result, review);
    console.log(`[Dye] review ${review} (base | mask: magenta accent, cyan extended | repainted base)`);
    if (args.includes('--render3d')) {
      const { findBlender, TOOLKIT_ROOT } = await import('./blender.mjs');
      const blender = findBlender();
      if (!blender) throw new Error('--render3d needs Blender (set BLENDER_PATH)');
      const sharp = await sharpLib();
      const stem = review.replace(/\.png$/i, '');
      const overlay = `${stem}_overlay.png`;
      const { width, height } = result.review;
      await sharp(Buffer.from(result.review.overlay), { raw: { width, height, channels: 4 } }).png().toFile(overlay);
      const out3d = `${stem}_3d.png`;
      const r = spawnSync(blender, ['-b', '--factory-startup', '-P', path.join(TOOLKIT_ROOT, 'scripts/blender/dye_render_review.py'), '--',
        path.resolve(output), path.resolve(overlay), path.resolve(out3d)], { stdio: 'inherit' });
      if (r.status !== 0) throw new Error('Blender review render failed');
      const base = out3d.replace(/\.png$/i, '');
      const views = [0, 1, 2, 3].map((i) => `${base}_v${i}.png`);
      await sharp({ create: { width: 512 * 4, height: 512, channels: 4, background: { r: 24, g: 24, b: 28, alpha: 1 } } })
        .composite(views.map((inp, i) => ({ input: inp, left: i * 512, top: 0 })))
        .png()
        .toFile(out3d);
      for (const v of views) fs.rmSync(v, { force: true });
      fs.rmSync(overlay, { force: true });
      console.log(`[Dye] 3D review ${out3d} (front, right, back, left; bind pose)`);
    }
  }
}

async function cmdBbox(args) {
  const input = arg(args, '--in');
  if (!input) throw new Error('bbox needs --in');
  const io = await createIO();
  const doc = await io.read(input);
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const p = prim.getAttribute('POSITION');
      if (!p) continue;
      console.log(`${mesh.getName() || 'mesh'}: min ${JSON.stringify(p.getMin([]))} max ${JSON.stringify(p.getMax([]))}`);
    }
  }
}

function cmdVerify(files) {
  let bad = 0;
  for (const f of files) {
    const res = checkDyeMaskJson(readGlbJson(new Uint8Array(fs.readFileSync(f))));
    console.log(`${res.ok ? 'OK  ' : 'FAIL'} ${f}${res.ok ? '' : `: ${res.errors.join('; ')}`}`);
    if (!res.ok) bad++;
  }
  if (bad) process.exit(1);
}

export async function runDyeMask(argv) {
  const [cmd, ...rest] = argv;
  if (!cmd || cmd === '--help' || cmd === '-h' || rest.includes('--help')) {
    console.log(HELP);
    return;
  }
  if (cmd === 'apply') await cmdApply(rest);
  else if (cmd === 'bbox') await cmdBbox(rest);
  else if (cmd === 'verify') cmdVerify(rest);
  else {
    console.error(`Unknown dye-mask command: ${cmd}`);
    console.log(HELP);
    process.exit(1);
  }
}
