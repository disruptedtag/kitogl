"""`GLVideoLayer`: its presentation clock (no card needed) and what it draws (needs an OpenGL surface).

The drawing tests skip where the Qt platform has no OpenGL widgets (Qt 6's `offscreen`). On Linux CI they run under Xvfb
with Mesa's software OpenGL, which the layer accepts only because the tests allow it (`allow_software_renderer=True`).
"""

from __future__ import annotations

import ctypes
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from kitogl import _qt
from kitogl.gl import platform_can_host_gl_widget
from kitogl.layer import GLVideoLayer

QtGui = importlib.import_module(_qt.QT_API + ".QtGui")
QtWidgets = importlib.import_module(_qt.QT_API + ".QtWidgets")


class PresentationClockTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows native swap control")
    def test_high_refresh_output_waits_one_physical_refresh(self) -> None:
        calls = []
        context = _qt.QOpenGLContext.currentContext()
        state = SimpleNamespace(_ready=True, _display_key=None, _presentation_rate=60.0, _vsync=True,
                                _refresh_hz=60.0, _swap_interval=1, _last_swap=None, _refresh_steps=0,
                                _fast_swaps=0, context=lambda: context,
                                screen=lambda: SimpleNamespace(refreshRate=lambda: 180.0),
                                _swap_control=lambda value: calls.append(value) or True)
        GLVideoLayer._configure_presentation(state)
        self.assertEqual(calls, [1], "video cadence must not add a three-refresh driver wait")

    def test_timer_pacing_uses_video_cadence_until_there_is_a_mix(self) -> None:
        state = SimpleNamespace(_refresh_hz=180.0, _timer_rate=0.0, _scene_clock=None,
                                _presentation_rate=24.0, _output_fps=0, _apply_frame_rate=mock.Mock())
        GLVideoLayer._pace_with_timer(state)
        self.assertEqual(state._timer_rate, 24)
        GLVideoLayer._display_refreshes(state, 0)
        for index in range(1, 8):
            jitter = 0.0015 if index % 2 else -0.0015
            self.assertEqual(GLVideoLayer._display_refreshes(state, index / 24 + jitter), 7.5)
        state._scene_clock = lambda: 0.5
        GLVideoLayer._pace_with_timer(state)
        self.assertEqual(state._timer_rate, 180)
        state._output_fps = 60
        GLVideoLayer._pace_with_timer(state)
        self.assertEqual(state._timer_rate, 60)

    def test_display_refresh_clock_counts_elapsed_time_without_callback_drift(self) -> None:
        state = SimpleNamespace(_refresh_epoch=None, _refresh_count=0, _refresh_hz=180.0, _timer_rate=0.0)
        self.assertEqual(GLVideoLayer._display_refreshes(state, 0.0), 1)
        ticks = [tick for tick in range(1, 180) if tick % 17 != 0]
        times = [(tick + (0.35 if tick % 2 else -0.35)) / 180 for tick in ticks] + [1.0]
        steps = [GLVideoLayer._display_refreshes(state, now) for now in times]
        self.assertEqual(sum(steps), 180, "late callbacks must not lose refreshes across a full second")
        self.assertGreater(max(steps), 1, "the omitted refresh callbacks still count")
        self.assertEqual(GLVideoLayer._display_refreshes(state, 1.0001), 0,
                         "an extra paint in the same refresh must not advance the video")


class _Plane:
    """One 8-bit plane in memory the card can read by address."""

    def __init__(self, value: int, columns: int, rows: int) -> None:
        self._buffer = ctypes.create_string_buffer(bytes([value]) * (columns * rows), columns * rows)
        self.buffer_ptr = ctypes.addressof(self._buffer)


class _FlatPicture:
    """A 4:2:0 picture of one colour, with padded rows like FFmpeg's."""

    def __init__(self, width: int, height: int, y: int, u: int, v: int, matrix: str = "601",
                 full_range: bool = False, padding: int = 0) -> None:
        self.width, self.height, self.matrix, self.full_range = width, height, matrix, full_range
        chroma = ((width + 1) // 2 + padding, (height + 1) // 2)
        self.line_sizes = ((width + padding, height), chroma, chroma)
        self.planes = (_Plane(y, width + padding, height), _Plane(u, *chroma), _Plane(v, *chroma))


def _bt601_limited(y: int, u: int, v: int):
    luma = (y - 16) * 255.0 / 219.0
    cb, cr = (u - 128) * 255.0 / 224.0, (v - 128) * 255.0 / 224.0
    rgb = (luma + 1.4020 * cr, luma - 0.3441 * cb - 0.7141 * cr, luma + 1.7720 * cb)
    return tuple(int(round(min(255.0, max(0.0, channel)))) for channel in rgb)


class DrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        if not platform_can_host_gl_widget():
            raise unittest.SkipTest("no OpenGL widget on this Qt platform")

    def _render(self, picture, width: int, height: int, overlay=None):
        parent = QtWidgets.QWidget()
        parent.resize(width, height)
        layer = GLVideoLayer(parent, allow_software_renderer=True)
        layer.setGeometry(0, 0, width, height)
        layer.set_picture_source(lambda: picture)
        if overlay is not None:
            layer.set_text(*overlay)
        parent.show()
        layer.show()
        layer.update()
        self.app.processEvents()
        image = layer.grabFramebuffer()
        broken, drawn = layer.broken, layer.frames_drawn
        layer.release()
        layer.close()
        layer.deleteLater()
        parent.close()
        if broken:
            self.skipTest("this OpenGL cannot run the layer")
        self.assertGreater(drawn, 0)
        return image

    def _assert_colour(self, image, x: int, y: int, expected, tolerance: int = 3) -> None:
        colour = image.pixelColor(x, y)
        got = (colour.red(), colour.green(), colour.blue())
        self.assertTrue(all(abs(a - b) <= tolerance for a, b in zip(got, expected)), f"{got} != {expected} at {x},{y}")

    def test_a_601_red_is_converted_and_shrunk_2_to_1(self) -> None:
        image = self._render(_FlatPicture(320, 180, 81, 90, 240), 160, 90)
        expected = _bt601_limited(81, 90, 240)
        for x, y in ((1, 1), (80, 45), (158, 88)):
            self._assert_colour(image, x, y, expected)

    def test_padded_rows_are_not_drawn(self) -> None:
        image = self._render(_FlatPicture(200, 100, 145, 54, 34, padding=56), 200, 100)
        self._assert_colour(image, 198, 50, _bt601_limited(145, 54, 34))

    def test_the_overlay_goes_where_its_ink_is(self) -> None:
        width, height = 160, 90
        overlay = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32_Premultiplied)
        overlay.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(overlay)
        painter.fillRect(_qt.QRect(20, 10, 40, 20), QtGui.QColor(255, 255, 255))
        painter.end()
        grey = _FlatPicture(width, height, 126, 128, 128)
        image = self._render(grey, width, height, overlay=(overlay, _qt.QRect(20, 10, 40, 20)))
        self._assert_colour(image, 40, 20, (255, 255, 255))
        self._assert_colour(image, 100, 60, _bt601_limited(126, 128, 128))


if __name__ == "__main__":
    unittest.main()
