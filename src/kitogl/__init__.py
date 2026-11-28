"""KitoGL: an OpenGL 2.1 / ES 2 video compositor for PySide6 and PySide2.

YUV to RGB in one shader, "cover" cropping, box-filtered shrinking, an RGBA overlay and scene transitions (fade, wipe,
push), on a native child surface of a Qt window. Pure Python: one wheel for every platform, Python 3.8 and newer.

`kitogl.color`, `kitogl.geometry` and `kitogl.shaders` need no Qt. `GLVideoLayer` and `FrameUpdates` load the binding the
application already uses (see `kitogl._qt`) the first time they are asked for.
"""

from __future__ import annotations

__version__ = "0.1.1"
__all__ = ["FrameUpdates", "GLVideoLayer", "__version__"]


def __getattr__(name: str):
    if name == "GLVideoLayer":
        from kitogl.layer import GLVideoLayer

        return GLVideoLayer
    if name == "FrameUpdates":
        from kitogl.frame_updates import FrameUpdates

        return FrameUpdates
    raise AttributeError(f"module 'kitogl' has no attribute {name!r}")
