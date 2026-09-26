/**
 * Cost calculation and estimation module for Shards of Stone 3D Asset Toolkit.
 */

import { parseArgs } from 'node:util';

export const PRICING_TABLE = {
  // OpenAI gpt-image-2.5 (Flare / Sunburst share one token rate card: $30 per 1M
  // image output tokens). Approximate USD per 1024x1024 image by quality, from
  // OpenAI's calculator (Sep 2026); medium and xhigh are interpolated. The
  // concept step logs the real cost of every call from the API's usage field.
  // Note: 2.5's "high" spends what gpt-image-2's "medium" did.
  openai: {
    'gpt-image-2.5': {
      label: 'OpenAI gpt-image-2.5 (Recommended for concepts, quality high)',
      costPerImageByQuality: { low: 0.006, medium: 0.011, high: 0.053, xhigh: 0.11, max: 0.211 },
      defaultQuality: 'high',
    },
  },
  gemini: {
    'gemini-2.5-flash-image': {
      label: 'Gemini 2.5 Flash Image',
      costPerImage: 0.039,
      inputPerMTok: 0.30,
    },
    'gemini-3.1-flash-image': {
      label: 'Gemini 3.1 Flash Image',
      costPerImage: 0.067,
      inputPerMTok: 0.50,
    },
    'gemini-3-pro-image': {
      label: 'Gemini 3 Pro Image (Nano Banana Pro)',
      costPerImage: 0.134,
      inputPerMTok: 2.00,
    },
  },
  meshy: {
    creditCostUsd: 0.012, // approx $12 per 1,000 credits on standard plan
    actions: {
      imageTo3d: 30,      // 30 credits
      retexture: 10,      // 10 credits
      autoRigHumanoid: 10, // 10 credits
      remesh: 5           // 5 credits per LOD rung (re-bakes the texture onto the new mesh)
    }
  },
  llmAgent: {
    'claude-3-5-sonnet': {
      label: 'Claude 3.5 Sonnet / Astra',
      inputPerMTok: 3.00,
      outputPerMTok: 15.00,
      typicalTokensPerUnit: 25000,
    },
    'claude-opus-5': {
      label: 'Claude Opus 5 / Expert Rigging',
      inputPerMTok: 15.00,
      outputPerMTok: 75.00,
      typicalTokensPerUnit: 25000,
    }
  }
};

export function calculateUnitEstimate(options = {}) {
  const {
    conceptModel = 'gpt-image-2.5',
    conceptQuality = 'high',
    conceptCandidates = 3,
    useAgent = true,
    agentModel = 'claude-3-5-sonnet',
    meshyAction = 'imageTo3d',
  } = options;

  const openaiTier = PRICING_TABLE.openai[conceptModel];
  const geminiTier = PRICING_TABLE.gemini[conceptModel];
  if (!openaiTier && !geminiTier) {
    throw new Error(`Unknown concept model "${conceptModel}". Known: ${[...Object.keys(PRICING_TABLE.openai), ...Object.keys(PRICING_TABLE.gemini)].join(', ')}`);
  }
  const perImage = openaiTier
    ? (openaiTier.costPerImageByQuality[conceptQuality] ?? openaiTier.costPerImageByQuality[openaiTier.defaultQuality])
    : geminiTier.costPerImage;
  const conceptCost = perImage * conceptCandidates;

  const meshyCredits = PRICING_TABLE.meshy.actions[meshyAction] || 30;
  const meshyCost = meshyCredits * PRICING_TABLE.meshy.creditCostUsd;

  let agentCost = 0;
  if (useAgent) {
    const agentTier = PRICING_TABLE.llmAgent[agentModel] || PRICING_TABLE.llmAgent['claude-3-5-sonnet'];
    const inputRatio = 0.7;
    const outputRatio = 0.3;
    const tokens = agentTier.typicalTokensPerUnit;
    const inputTokens = tokens * inputRatio;
    const outputTokens = tokens * outputRatio;

    agentCost = (inputTokens / 1_000_000) * agentTier.inputPerMTok +
                (outputTokens / 1_000_000) * agentTier.outputPerMTok;
  }

  const total = conceptCost + meshyCost + agentCost;

  return {
    conceptCost,
    conceptCandidates,
    meshyCredits,
    meshyCost,
    agentCost,
    totalCost: total,
    localComputeCost: 0.00
  };
}

export async function runEstimateCost(args) {
  const { values } = parseArgs({
    args,
    options: {
      candidates: { type: 'string', default: '3' },
      model: { type: 'string', default: 'gpt-image-2.5' },
      quality: { type: 'string', default: 'high' },
      agent: { type: 'string', default: 'claude-3-5-sonnet' },
      units: { type: 'string', default: '1' }
    }
  });

  const unitCount = parseInt(values.units, 10) || 1;
  const candidates = parseInt(values.candidates, 10) || 3;

  const estimate = calculateUnitEstimate({
    conceptModel: values.model,
    conceptQuality: values.quality,
    conceptCandidates: candidates,
    agentModel: values.agent,
  });

  console.log('\n=== Shards of Stone 3D Asset Generation Cost Estimate ===\n');
  console.log(`Per-unit breakdown (${candidates} concept candidates, 30k poly mesh, auto-rigging):`);
  console.log(`  1. Concept Art (${values.model}): $${estimate.conceptCost.toFixed(3)}  (${candidates} candidates at ~$${(estimate.conceptCost / candidates).toFixed(3)} each)`);
  console.log(`  2. 3D Mesh Synthesis (Meshy): $${estimate.meshyCost.toFixed(3)}  (${estimate.meshyCredits} credits)`);
  console.log(`  3. Rigging & Analysis (Opus): $${estimate.agentCost.toFixed(3)}  (~25k tokens for landmark & weight validation)`);
  console.log(`  4. Local Blender & VAT bake:  $0.000  (Free local compute)`);
  console.log('  ------------------------------------------------');
  console.log(`  Estimated Total per Unit:     $${estimate.totalCost.toFixed(2)} USD\n`);

  if (unitCount > 1) {
    console.log(`Total for batch of ${unitCount} units:`);
    console.log(`  Total Estimated Cost:         $${(estimate.totalCost * unitCount).toFixed(2)} USD`);
    console.log(`  Total Meshy Credits:          ${estimate.meshyCredits * unitCount} credits\n`);
  }
}
