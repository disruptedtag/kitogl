"""The parts of KitoGL that need no Qt and no graphics card."""

from __future__ import annotations

import sys
import unittest

from kitogl import color, geometry, shaders


class GeometryTests(unittest.TestCase):
    def test_cover_keeps_everything_when_the_shapes_match(self) -> None:
        self.assertEqual(geometry.cover_geometry(1920, 1080, 960, 540)[:4], (0.0, 0.0, 1.0, 1.0))

    def test_cover_crops_the_sides_of_a_wide_picture_on_a_narrow_screen(self) -> None:
        u0, v0, u1, v1, ratio = geometry.cover_geometry(1920, 1080, 800, 600)
        self.assertEqual((v0, v1), (0.0, 1.0))  # the height fits: only the sides are cut
        self.assertGreater(u0, 0.0)
        self.assertAlmostEqual(u0 + u1, 1.0)
        self.assertAlmostEqual(ratio, 1080 / 600)

    def test_taps_follow_the_shrink(self) -> None:
        self.assertEqual([geometry.taps_for(r) for r in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)], [1, 1, 1, 1, 2, 2, 4])

    def test_easing_starts_and_ends_at_rest(self) -> None:
        self.assertEqual((geometry.eased(-1.0), geometry.eased(0.0), geometry.eased(0.5), geometry.eased(1.0),
                          geometry.eased(2.0)), (0.0, 0.0, 0.5, 1.0, 1.0))
        self.assertEqual(sorted(geometry.DIRECTIONS), ["down", "left", "right", "up"])


class ShaderTests(unittest.TestCase):
    def test_the_shader_is_straight_line_code_with_one_sample_per_tap(self) -> None:
        for taps in (1, 2, 3, 4):
            source = shaders.yuv_source(taps)
            self.assertNotIn("for (", source)  # no loop, no `break`: GLSL ES 1.00 (ANGLE) is strict about them
            self.assertNotIn("break", source)
            self.assertEqual(source.count("sum += yuv("), taps * taps)
            self.assertIn(f"sum / {taps * taps:.1f}", source)
        self.assertEqual(shaders.yuv_source(9).count("sum += yuv("), 16)  # never more than 4 per axis
        self.assertEqual(shaders.yuv_source(0).count("sum += yuv("), 1)

    def test_every_source_stays_valid_for_gl_2_1_and_es_2(self) -> None:
        import re

        for source in (shaders.VERTEX, shaders.RGBA, shaders.SCENE, shaders.yuv_source(2)):
            code = re.sub(r"//[^\n]*", "", source)
            self.assertNotIn("#version", code)
            self.assertNotIn("texture(", code.replace("texture2D(", ""))  # GLSL 1.30+ only
            self.assertIsNone(re.search(r"(^|[;{]\s*)(in|out)\s", code, re.MULTILINE))  # 1.30+ qualifiers

    def test_every_matrix_has_its_four_coefficients(self) -> None:
        self.assertEqual(set(color.COEFFICIENTS), {color.MATRIX_601, color.MATRIX_709, color.MATRIX_2020})
        self.assertTrue(all(len(values) == 4 for values in color.COEFFICIENTS.values()))

    def test_these_modules_do_not_load_qt(self) -> None:
        import subprocess

        code = ("import sys, kitogl, kitogl.color, kitogl.geometry, kitogl.shaders, kitogl.trace, kitogl.picture, kitogl.gl; "
                "print(sorted(name for name in ('PySide6', 'PySide2', 'kitogl._qt') if name in sys.modules))")
        output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        self.assertEqual(output.strip(), "[]")


class _Program:
    """Counts `uniformLocation` queries and records what was set, like a linked QOpenGLShaderProgram."""

    def __init__(self) -> None:
        self.queries, self.set = [], []

    def uniformLocation(self, name):  # noqa: N802 - Qt's name
        self.queries.append(name)
        return {"amount": 3, "tex": 5}.get(name, -1)

    def setUniformValue(self, location, x, y):  # noqa: N802
        self.set.append((location, x, y))

    def setUniformValue1i(self, location, value):  # noqa: N802
        self.set.append((location, value))

    def setUniformValue1f(self, location, value):  # noqa: N802
        self.set.append((location, value))


class UniformTests(unittest.TestCase):
    def test_a_location_is_asked_once_per_program_and_name(self) -> None:
        # Regression (0.1.1): asked on every frame, the location query kept a whole CPU core busy at 180 Hz.
        from kitogl.gl import set_uniform

        first, second = _Program(), _Program()
        for _ in range(3):
            set_uniform(first, "amount", 0.5)
            set_uniform(first, "tex", 1)
        set_uniform(first, "missing", 1.0, 2.0)
        set_uniform(first, "missing", 1.0, 2.0)
        set_uniform(second, "amount", 0.25)
        self.assertEqual(first.queries, ["amount", "tex", "missing"])
        self.assertEqual(second.queries, ["amount"])  # another program: its own locations
        self.assertEqual(first.set[:2], [(3, 0.5), (5, 1)])
        self.assertEqual(first.set[-1], (-1, 1.0, 2.0))  # a uniform the shader optimised away stays a no-op


if __name__ == "__main__":
    unittest.main()
