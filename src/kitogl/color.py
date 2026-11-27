"""Colour matrices the YUV shader uses. No Qt.

A picture names its matrix with one of the `MATRIX_*` strings; the shader multiplies the centred chroma by these numbers.
"""

from __future__ import annotations

MATRIX_709, MATRIX_601, MATRIX_2020 = "709", "601", "2020"

# (red from V, green from U, green from V, blue from U): what the shader multiplies the centred chroma by.
COEFFICIENTS = {
    MATRIX_709: (1.5748, 0.1873, 0.4681, 1.8556),
    MATRIX_601: (1.4020, 0.3441, 0.7141, 1.7720),
    MATRIX_2020: (1.4746, 0.1646, 0.5714, 1.8814),
}
