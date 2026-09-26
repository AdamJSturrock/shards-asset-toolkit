/**
 * Concept generation module.
 * Enforces orthographic 3/4 camera angles, clean backgrounds, and effect stripping.
 *
 * Providers:
 *   - openai (default when OPENAI_API_KEY is set): gpt-image-2.5 at quality "high".
 *     gpt-image-2.5-flare for text-only prompts, gpt-image-2.5-sunburst when a
 *     reference image is attached (--ref), via the edits endpoint.
 *   - gemini (default otherwise): Gemini 3 Pro Image ("Nano Banana Pro").
 *
 * Quality note: 2.5's "high" spends what gpt-image-2's "medium" did. "max"
 * costs more per image than Gemini 3 Pro Image, so it is never the default.
 */

import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';

const GEMINI_IMAGE_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent';
const OPENAI_BASE = 'https://api.openai.com/v1';

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

export function resolveProvider(explicit) {
  if (explicit === 'openai' || explicit === 'gemini') return explicit;
  return process.env.OPENAI_API_KEY ? 'openai' : 'gemini';
}

async function generateOpenAI({ prompt, refPaths, quality, size, model }) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('OPENAI_API_KEY is missing. Get one at https://platform.openai.com/ and set it in your .env');
  let res;
  if (refPaths.length > 0) {
    const form = new FormData();
    form.append('model', model || 'gpt-image-2.5-sunburst');
    form.append('prompt', prompt);
    form.append('size', size);
    form.append('quality', quality);
    form.append('output_format', 'png');
    for (const p of refPaths) {
      form.append('image[]', new Blob([fs.readFileSync(p)], { type: 'image/png' }), path.basename(p));
    }
    res = await fetch(`${OPENAI_BASE}/images/edits`, { method: 'POST', headers: { Authorization: `Bearer ${key}` }, body: form });
  } else {
    res = await fetch(`${OPENAI_BASE}/images/generations`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: model || 'gpt-image-2.5-flare', prompt, size, quality, output_format: 'png', n: 1 }),
    });
  }
  if (!res.ok) throw new Error(`OpenAI image API error (${res.status}): ${(await res.text()).slice(0, 500)}`);
  const json = await res.json();
  const b64 = json.data?.[0]?.b64_json;
  if (!b64) throw new Error('OpenAI returned no image payload');
  if (json.usage) {
    const d = json.usage.input_tokens_details || {};
    const usd = ((d.text_tokens || 0) * 5 + (d.image_tokens || 0) * 8 + (json.usage.output_tokens || 0) * 30) / 1e6;
    console.log(`[Concept] OpenAI cost for this image: ~$${usd.toFixed(4)}`);
  }
  return Buffer.from(b64, 'base64');
}

async function generateGemini({ prompt, refPaths }) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is missing. Get one at https://aistudio.google.com/ and set it in your .env');
  const parts = [{ text: prompt }];
  for (const p of refPaths) parts.push({ inline_data: { mime_type: 'image/png', data: fs.readFileSync(p).toString('base64') } });
  const response = await fetch(`${GEMINI_IMAGE_ENDPOINT}?key=${apiKey}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // responseModalities must include IMAGE or the model answers in text only.
    body: JSON.stringify({ contents: [{ parts }], generationConfig: { responseModalities: ['TEXT', 'IMAGE'] } }),
  });
  if (!response.ok) throw new Error(`Gemini API error (${response.status}): ${(await response.text()).slice(0, 500)}`);
  const data = await response.json();
  const candidate = data.candidates?.[0];
  const imagePart = candidate?.content?.parts?.find((p) => p.inlineData);
  if (!imagePart?.inlineData?.data) {
    const text = candidate?.content?.parts?.[0]?.text;
    throw new Error(`Gemini returned no image payload${text ? `: ${text.slice(0, 200)}` : ''}`);
  }
  return Buffer.from(imagePart.inlineData.data, 'base64');
}

export async function runConcept(args) {
  const { values } = parseArgs({
    args,
    options: {
      prompt: { type: 'string' },
      output: { type: 'string' },
      candidates: { type: 'string', default: '1' },
      'strip-effects': { type: 'boolean', default: true },
      provider: { type: 'string' },              // openai | gemini
      quality: { type: 'string', default: 'high' }, // openai only: low | medium | high | xhigh | max
      size: { type: 'string', default: '1024x1024' },
      model: { type: 'string' },
      ref: { type: 'string', multiple: true, default: [] }, // reference image(s)
    }
  });

  if (!values.prompt) {
    console.error('Error: --prompt is required');
    process.exit(1);
  }

  const provider = resolveProvider(values.provider);
  const outputPath = values.output || './output/concept.png';
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const enhancedPrompt = `${CONCEPT_SYSTEM_DIRECTIVE}\nSubject: ${values.prompt}`;
  const count = Math.max(1, parseInt(values.candidates, 10) || 1);

  console.log(`[Concept] Provider: ${provider}${provider === 'openai' ? ` (quality ${values.quality}, ${values.size})` : ' (Gemini 3 Pro Image)'}`);
  console.log(`[Concept] Prompt: "${values.prompt}"`);
  console.log(`[Concept] Effect stripping: ${values['strip-effects'] ? 'ENABLED' : 'DISABLED'}`);

  try {
    for (let i = 0; i < count; i++) {
      const buffer = provider === 'openai'
        ? await generateOpenAI({ prompt: enhancedPrompt, refPaths: values.ref, quality: values.quality, size: values.size, model: values.model })
        : await generateGemini({ prompt: enhancedPrompt, refPaths: values.ref });
      const out = count === 1 ? outputPath : outputPath.replace(/(\.png)?$/, `_${i}.png`);
      fs.writeFileSync(out, buffer);
      console.log(`[Concept] Saved concept to: ${out}`);
    }
  } catch (err) {
    console.error(`[Concept] Generation failed: ${err.message}`);
    process.exit(1);
  }
}
