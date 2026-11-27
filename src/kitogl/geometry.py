"""Where a picture goes on the screen and how hard it is shrunk, and the motion of the scene transitions. No Qt."""

from __future__ import annotations

import math
from typing import Dict, Tuple

# The four directions a wipe or a push can take: the edge (or the travel) goes towards this side.
DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
}


def eased(t: float) -> float:
    """Smoothstep: starts and ends at rest, so a fade does not snap into motion at either end."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def cover_geometry(picture_w: int, picture_h: int, target_w: int, target_h: int):
    """(u0, v0, u1, v1, ratio): the part of the picture that stays visible when it covers the target, and how many
    source pixels lie under one output pixel (more than 1 = shrinking)."""
    scale = max(target_w / picture_w, target_h / picture_h)
    visible_x = target_w / (picture_w * scale)
    visible_y = target_h / (picture_h * scale)
    u0, v0 = (1.0 - visible_x) / 2.0, (1.0 - visible_y) / 2.0
    return u0, v0, 1.0 - u0, 1.0 - v0, 1.0 / scale


def taps_for(ratio: float) -> int:
    """Bilinear samples per axis for a shrink of `ratio`: each one already averages two source pixels, so a 2:1
    shrink needs one (placed between the pixels), 4:1 needs two, and enlarging needs one."""
    return int(min(4, max(1, math.ceil(ratio / 2.0 - 0.01))))
