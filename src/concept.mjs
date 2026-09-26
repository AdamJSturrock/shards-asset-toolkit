/**
 * Concept generation module using Google Gemini Image API.
 * Enforces orthographic 3/4 camera angles, clean backgrounds, and effect stripping.
 */

import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';

const GEMINI_IMAGE_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent';

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
      candidates: { type: 'string', default: '1' },
      'strip-effects': { type: 'boolean', default: true },
    }
  });

  if (!values.prompt) {
    console.error('Error: --prompt is required');
    process.exit(1);
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

  const enhancedPrompt = `${CONCEPT_SYSTEM_DIRECTIVE}\nSubject: ${values.prompt}`;

  console.log(`[Concept] Generating concept via Gemini API...`);
  console.log(`[Concept] Prompt: "${values.prompt}"`);
  console.log(`[Concept] Effect stripping: ${values['strip-effects'] ? 'ENABLED' : 'DISABLED'}`);

  const url = `${GEMINI_IMAGE_ENDPOINT}?key=${apiKey}`;

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
      headers: { 'Content-Type': 'application/json' },
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
