// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * The owner-colour ramp and the unit dye mask's single-channel encoding, as
 * the Shards of Stone runtime uses them (CPU mirrors of its shader code).
 * Everything is sRGB 0..1.
 *
 * Mask encoding, one channel:  0.0 none · 0.5 extended (big cloth) · 1.0 accent.
 * ONE channel because the game's texture pipeline stores an occlusion texture
 * in KTX2 as R-only: a mask spread over G or B would be dropped.
 */

/** Owner colours brighter than this are scaled down before painting a BIG area. */
export const BIG_AREA_MAX_LUMA = 0.58;
export const BIG_AREA_HIGHLIGHT = 0.14;
export const CLOTH_HIGHLIGHT = 0.28;
/** Where the owner colour itself sits on the ramp. */
export const RAMP_MID = 0.55;
/** Floor of a measured value range, so a flat-shaded area cannot divide by ~0. */
export const MIN_LUM_SPAN = 0.05;

const mix = (a, b, t) => a + (b - a) * t;

/** [hue 0..1, saturation, value] */
export function dyeHsv(r, g, b) {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const d = mx - mn;
  let h = 0;
  if (d > 1e-5) {
    if (mx === r) h = ((((g - b) / d) % 6) + 6) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
  }
  return [h, mx > 1e-5 ? d / mx : 0, mx];
}

/** Place a texel's HSV value within a measured [lo, hi] range. */
export function dyeShade(value, lo, hi) {
  const t = (value - lo) / Math.max(hi - lo, MIN_LUM_SPAN);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Shaded ramp of an owner colour for small heraldic areas (banners, sashes). */
export function dyeRampCloth(c, t, out = [0, 0, 0]) {
  for (let i = 0; i < 3; i++) {
    const shadow = mix(c[i] * 0.32, c[i] * c[i] * 0.5, 0.5);
    const lit = mix(c[i], 1, CLOTH_HIGHLIGHT);
    out[i] = t < RAMP_MID ? mix(shadow, c[i], t / RAMP_MID) : mix(c[i], lit, (t - RAMP_MID) / (1 - RAMP_MID));
  }
  return out;
}

/** The luminance-capped colour for a big area (a cloak, a caparison). */
export function dyeBigAreaColor(c, out = [0, 0, 0]) {
  const l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  const k = Math.min(1, BIG_AREA_MAX_LUMA / Math.max(l, 1e-4));
  out[0] = c[0] * k;
  out[1] = c[1] * k;
  out[2] = c[2] * k;
  return out;
}

/** Big-area ramp. `c` must already be capped with dyeBigAreaColor. */
export function dyeRampBigArea(c, t, out = [0, 0, 0]) {
  for (let i = 0; i < 3; i++) {
    const shadow = mix(c[i] * 0.38, c[i] * c[i] * 0.55, 0.5);
    const lit = mix(c[i], 1, BIG_AREA_HIGHLIGHT);
    out[i] = t < RAMP_MID ? mix(shadow, c[i], t / RAMP_MID) : mix(c[i], lit, (t - RAMP_MID) / (1 - RAMP_MID));
  }
  return out;
}

/** 0xRRGGBB -> sRGB 0..1 triple. */
export function hexToRgb01(hex, out = [0, 0, 0]) {
  out[0] = ((hex >> 16) & 0xff) / 255;
  out[1] = ((hex >> 8) & 0xff) / 255;
  out[2] = (hex & 0xff) / 255;
  return out;
}

/** Encode (accent, extended) weights, 0..1, into the mask level. */
export function encodeUnitDyeMask(accent, extended) {
  if (accent > 0) return 0.5 + 0.5 * Math.min(1, accent);
  return 0.5 * Math.min(1, Math.max(0, extended));
}

/** Mask level -> [accent, extended] weights. */
export function decodeUnitDyeMask(v) {
  const acc = Math.min(1, Math.max(0, (v - 0.5) * 2));
  const ext = v <= 0.5 ? Math.min(1, Math.max(0, v * 2)) : 1 - acc;
  return [acc, ext];
}
