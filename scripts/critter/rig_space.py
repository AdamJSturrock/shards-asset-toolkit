# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""The rotation from a Meshy import (Blender Z-up) to rig space, shared by
prep_critter.py, rig_critter.py and add_limb.py so every tool puts a vertex
in the same place: pitch about X first (a model authored standing up, like
the heraldic drakeling, is laid into its flight pose), then yaw about Z.
Row-vector convention: rig = co @ rig_rotation(yaw, pitch)."""
import numpy as np


def rig_rotation(yaw_deg=0.0, pitch_deg=0.0):
    a = np.radians(yaw_deg); c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
    b = np.radians(pitch_deg); cb, sb = np.cos(b), np.sin(b)
    Rx = np.array([[1, 0, 0], [0, cb, sb], [0, -sb, cb]], dtype=np.float32)
    return Rx @ Rz
