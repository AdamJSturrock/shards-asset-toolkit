// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Credit and API cost estimate for a batch of units. Only prices we have
 * measured or that the vendor lists are used; anything that depends on how a
 * run goes (retries, extra concept candidates, agent iteration) is left for
 * you to add, and the report says so.
 */

import { parseArgs } from 'node:util';

export const PRICING = {
  gemini: {
    // about $0.134 per 1K/2K image
    'gemini-3-pro-image-preview': { label: 'Gemini 3 Pro Image', usdPerImage: 0.134 },
  },
  meshy: {
    imageTo3d: 30, // credits per image-to-3D mesh
    autoRigHumanoid: 5, // credits per humanoid auto-rig (no animations)
  },
};

export function calculateUnitEstimate({ candidates = 3, meshyRig = true, usdPerCredit = null } = {}) {
  const tier = PRICING.gemini['gemini-3-pro-image-preview'];
  const conceptUsd = tier.usdPerImage * candidates;
  const meshyCredits = PRICING.meshy.imageTo3d + (meshyRig ? PRICING.meshy.autoRigHumanoid : 0);
  const meshyUsd = usdPerCredit == null ? null : meshyCredits * usdPerCredit;
  return { conceptUsd, candidates, meshyCredits, meshyUsd };
}

const HELP = `shards-asset estimate-cost: API credits and cost for a batch of units

Usage:
  shards-asset estimate-cost [--units 1] [--candidates 3] [--no-meshy-rig] [--usd-per-credit 0.02]

Priced items:
  Gemini 3 Pro Image      ~$0.134 per concept image
  Meshy image-to-3D       30 credits per mesh
  Meshy humanoid auto-rig 5 credits (skip with --no-meshy-rig for creatures and
                          for humanoids Meshy refuses, rigged in Blender instead)
  Blender, Mixamo, glTF-Transform: local and free

Meshy credits are converted to dollars only if you pass --usd-per-credit (it
depends on your plan). Retries, rejected meshes and any coding-agent tokens
are not included: they depend on the unit.`;

export async function runEstimateCost(args) {
  const { values } = parseArgs({
    args,
    options: {
      units: { type: 'string', default: '1' },
      candidates: { type: 'string', default: '3' },
      'no-meshy-rig': { type: 'boolean', default: false },
      'usd-per-credit': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    console.log(HELP);
    return;
  }
  const units = parseInt(values.units, 10) || 1;
  const candidates = parseInt(values.candidates, 10) || 3;
  const usdPerCredit = values['usd-per-credit'] ? Number(values['usd-per-credit']) : null;
  const e = calculateUnitEstimate({ candidates, meshyRig: !values['no-meshy-rig'], usdPerCredit });

  console.log('\n=== Shards of Stone asset toolkit: cost estimate ===\n');
  console.log(`Per unit:`);
  console.log(`  Concept art (Gemini 3 Pro Image): $${e.conceptUsd.toFixed(3)}  (${candidates} images at ~$0.134)`);
  console.log(`  Meshy:                            ${e.meshyCredits} credits  (30 mesh${values['no-meshy-rig'] ? '' : ' + 5 auto-rig'})`
    + (e.meshyUsd == null ? '' : `  = $${e.meshyUsd.toFixed(2)} at $${usdPerCredit}/credit`));
  console.log('  Blender, Mixamo, glTF-Transform:   free (local)');
  if (units > 1) {
    console.log(`\nFor ${units} units: $${(e.conceptUsd * units).toFixed(2)} of Gemini images and ${e.meshyCredits * units} Meshy credits`
      + (e.meshyUsd == null ? '' : ` ($${(e.meshyUsd * units).toFixed(2)})`));
  }
  console.log('\nNot included: retries, rejected meshes, and coding-agent tokens if you use an agent to drive the rigging.\n');
}
