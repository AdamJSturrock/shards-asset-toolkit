# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Where the critter-authoring tools read and write their working files.

Profiles: scripts/critter/profiles.json, or the file named by
CRITTER_PROFILES (e.g. your own profiles next to your meshes).

Working files: ./output/critter-authoring/<critter>/ under the current
directory. Set CRITTER_AUTHORING_OUT to keep the heavy prep renders, .blend
masters and review video somewhere else, e.g.

  CRITTER_AUTHORING_OUT=~/critter-work python3 scripts/critter/fit_landmarks.py reef_crab
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Relative paths (a profile's `source`, the work dir) resolve against the
# directory the tools are run from, normally the toolkit root.
ROOT = os.getcwd()


def profiles_path():
    return os.path.expanduser(os.environ.get('CRITTER_PROFILES') or os.path.join(HERE, 'profiles.json'))


def load_profiles():
    return json.load(open(profiles_path()))


def work_dir(*parts):
    base = os.environ.get('CRITTER_AUTHORING_OUT') or os.path.join(ROOT, 'output', 'critter-authoring')
    return os.path.join(os.path.expanduser(base), *parts)


def source_glb(name, prof):
    """The static source mesh: the profile's `source` (absolute, or relative to
    the current directory; `~` expanded). Every step that reads the source
    also takes --input to override it."""
    src = prof.get('source')
    if not src:
        raise SystemExit(f'profile {name!r} has no "source"; set one in {profiles_path()} or pass --input')
    src = os.path.expanduser(src)
    return src if os.path.isabs(src) else os.path.join(ROOT, src)
