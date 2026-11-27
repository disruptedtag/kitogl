"""What the layer needs from a decoded picture. KitoGL does not decode: the application hands it objects like these.

Only 8-bit 4:2:0 reaches the card: three planes (Y, U, V) or, for a CUDA-resident frame, two (Y and interleaved UV).
Nothing is copied: each plane's `buffer_ptr` is where the card reads it, rows are `line_size` bytes (FFmpeg pads them;
the shader ignores the padding), and the picture must keep the memory alive while the layer may still upload it.
"""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    from typing import Protocol
except ImportError:  # Python 3.7: the contract is documentation only
    Protocol = object  # type: ignore[assignment,misc]


class Plane(Protocol):
    buffer_ptr: int  # address of the first byte


class Picture(Protocol):
    width: int
    height: int
    matrix: str  # one of `kitogl.color.MATRIX_*`
    full_range: bool  # True for JPEG / full-range video
    planes: Sequence[Plane]
    line_sizes: Sequence[Tuple[int, int]]  # per plane: (bytes in a row, rows)
    # Optional: `cuda_resident = True` on a frame that lives on an NVIDIA card (NV12); the layer then needs a
    # `cuda_uploader_factory` (see `GLVideoLayer`).
