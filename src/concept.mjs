// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
/**
 * Concept generation module using Google Gemini Image API.
 * Enforces orthographic 3/4 camera angles, clean backgrounds, and effect stripping.
 */

import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';

const GEMINI_API = 'https://generativelanguage.googleapis.com/v1beta/models';
// Gemini 3 Pro Image: about $0.134 per image at 1K/2K (Google list price).
const DEFAULT_MODEL = 'gemini-3-pro-image-preview';

const TPOSE_DIRECTIVE = `
POSE: a T-pose for rigging. Standing straight, legs slightly apart, both arms held straight out to the sides at shoulder height.
Any weapon stays gripped in its hand (do not drop it, do not open the hands).`;

const KEY_COLOUR_DIRECTIVE = `
PLAYER-COLOUR ZONES: paint small team-colour details (banners, plumes, sashes, trims) flat pure magenta #FF00FF,
and large team-colour cloth (a cloak, a tabard, a caparison) flat pure cyan #00FFFF. Use those two colours nowhere else.`;

const HELP = `shards-asset concept: generate a clean 2D concept image with Gemini

Usage:
  shards-asset concept --prompt "..." --output concepts/unit.png [--model ${DEFAULT_MODEL}] [--tpose] [--key-colours]

Options:
  --prompt <text>     subject description (required)
  --output <png>      default ./output/concept.png
  --model <id>        Gemini image model, default ${DEFAULT_MODEL}
  --tpose             ask for a T-pose with the weapon still in hand (for rigging humanoids;
                      do NOT use Meshy's pose_mode t-pose for armed units, it drops the weapon)
  --key-colours       paint team-colour zones magenta #FF00FF (accent) and cyan #00FFFF
                      (large cloth) for "shards-asset dye-mask"
  --no-strip-effects  keep particles/smoke in the prompt (not recommended for 3D)`;

export const CONCEPT_SYSTEM_DIRECTIVE = `
You are an expert game concept artist creating asset turnarounds for Shards of Stone, a fantasy RTS.
You must render:
- Single isolated subject in 3/4 isometric front view, full body visible from ground up.
- Crisp, distinct silhouette suitable for converting into a 3D polygonal model.
- Solid white or neutral monochrome background with no floor cast shadows or ground tiles.
- Hand-painted fantasy stylized textures with high contrast and readable material separation (chitin, rock, metal, cloth).
CRITICAL FOR 3D GENERATION:
- Do NOT draw particle effects, floating embers, magic swirls, fire trails, or smoke.
- Do NOT draw spider silk webs connecting limbs or dripping slime.
- Do NOT draw a decorative base, stone pedestal, or terrain patch under the character's feet.
`;

export async function runConcept(args) {
  const { values } = parseArgs({
    args,
    options: {
      prompt: { type: 'string' },
      output: { type: 'string' },
      model: { type: 'string', default: DEFAULT_MODEL },
      tpose: { type: 'boolean', default: false },
      'key-colours': { type: 'boolean', default: false },
      // accepted for backwards compatibility; effects are stripped unless --keep-effects
      'strip-effects': { type: 'boolean', default: true },
      'keep-effects': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    }
  });

  if (values.help || !values.prompt) {
    console.log(HELP);
    if (!values.help) process.exit(1);
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error('Error: GEMINI_API_KEY environment variable is missing.');
    console.error('Please obtain a key from https://aistudio.google.com/ and set it in your .env');
    process.exit(1);
  }

  const outputPath = values.output || './output/concept.png';
  const outDir = path.dirname(outputPath);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const directive = values['keep-effects']
    ? CONCEPT_SYSTEM_DIRECTIVE.split('CRITICAL FOR 3D GENERATION:')[0]
    : CONCEPT_SYSTEM_DIRECTIVE;
  const enhancedPrompt = `${directive}${values.tpose ? TPOSE_DIRECTIVE : ''}${values['key-colours'] ? KEY_COLOUR_DIRECTIVE : ''}\nSubject: ${values.prompt}`;

  console.log(`[Concept] Generating concept via ${values.model}...`);
  console.log(`[Concept] Prompt: "${values.prompt}"`);
  console.log(`[Concept] Effect stripping: ${values['keep-effects'] ? 'DISABLED' : 'ENABLED'} | T-pose: ${values.tpose} | key colours: ${values['key-colours']}`);

  const url = `${GEMINI_API}/${encodeURIComponent(values.model)}:generateContent`;

  const payload = {
    contents: [
      {
        parts: [
          { text: enhancedPrompt }
        ]
      }
    ],
    generationConfig: {
      temperature: 0.4,
    }
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      // the key goes in a header, never in the URL (URLs end up in logs)
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Gemini API error (${response.status}): ${errorText}`);
    }

    const data = await response.json();
    const candidate = data.candidates?.[0];
    const imagePart = candidate?.content?.parts?.find(p => p.inlineData);

    if (imagePart?.inlineData?.data) {
      const buffer = Buffer.from(imagePart.inlineData.data, 'base64');
      fs.writeFileSync(outputPath, buffer);
      console.log(`[Concept] Successfully saved clean concept to: ${outputPath}`);
    } else {
      console.log(`[Concept] Note: Model returned response without binary image payload.`);
      console.log(`[Concept] Check AI Studio console for Gemini multimodal image generation access.`);
      if (candidate?.content?.parts?.[0]?.text) {
        console.log(`[Concept] Response text: ${candidate.content.parts[0].text.slice(0, 200)}...`);
      }
    }
  } catch (err) {
    console.error(`[Concept] Generation failed: ${err.message}`);
    process.exit(1);
  }
}
