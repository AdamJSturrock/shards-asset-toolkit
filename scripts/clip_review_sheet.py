# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Compose contact sheets from scripts/blender/clip_review.py output.

One row per clip and view, labelled, plus a gameplay-size row per clip
(downscaled to --gameplay-px, the unit's on-screen height, then enlarged with
nearest-neighbour so the loss of detail stays visible).

    python3 scripts/clip_review_sheet.py --review-dir output/anim-authoring/goblin_worker/v1/review \
      --out output/anim-authoring/goblin_worker/v1/sheet.png --gameplay-px 64

Requires Pillow.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--review-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--clips', default='')
    parser.add_argument('--cell', type=int, default=192)
    parser.add_argument('--gameplay-px', type=int, default=64)
    args = parser.parse_args()
    root = Path(args.review_dir)
    meta = json.loads((root / 'frames.json').read_text())['clips']
    clips = [c for c in args.clips.split(',') if c] or list(meta)
    rows = []
    for clip in clips:
        files = sorted((root / clip).glob('*.png'))
        views = sorted({f.name.split('-')[0] for f in files})
        for view in views:
            rows.append((f'{clip} [{view}] {meta[clip]["duration"]}s', sorted((root / clip).glob(f'{view}-*.png')), False))
        if 'game' in views:
            rows.append((f'{clip} [game @ {args.gameplay_px}px]', sorted((root / clip).glob('game-*.png')), True))
    label_w, cell = 260, args.cell
    cols = max(len(r[1]) for r in rows)
    sheet = Image.new('RGB', (label_w + cols * cell, len(rows) * cell), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for y, (label, files, small) in enumerate(rows):
        draw.text((8, y * cell + cell // 2 - 6), label, fill=(230, 230, 230))
        for x, f in enumerate(files):
            im = Image.open(f).convert('RGB')
            if small:
                im = im.resize((args.gameplay_px, args.gameplay_px), Image.LANCZOS).resize((cell, cell), Image.NEAREST)
            else:
                im = im.resize((cell, cell), Image.LANCZOS)
            sheet.paste(im, (label_w + x * cell, y * cell))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f'wrote {args.out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
