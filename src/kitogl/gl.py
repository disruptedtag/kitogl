"""Small OpenGL helpers: is this a real card, can this Qt platform host a GL surface, and a uniform setter that works in
both bindings."""

from __future__ import annotations

# Names a software OpenGL goes by. On those, the shader runs on the CPU and a raster path is usually faster.
_SOFTWARE = ("llvmpipe", "softpipe", "swrast", "gdi generic", "microsoft basic", "software")
_PLATFORMS_WITHOUT_GL_WIDGET = ("offscreen", "minimal", "minimalegl", "linuxfb", "vnc")


def is_software_renderer(name: str) -> bool:
    lowered = (name or "").lower()
    return any(word in lowered for word in _SOFTWARE)


def platform_can_host_gl_widget() -> bool:
    """False on the Qt platforms that have no OpenGL widgets (`offscreen` of Qt 6, for example)."""
    from kitogl._qt import QGuiApplication

    return QGuiApplication.platformName().lower() not in _PLATFORMS_WITHOUT_GL_WIDGET


def set_uniform(program, name: str, *values) -> None:
    """A uniform by location, with the typed setters.

    The generic overloads misbehave in one binding or the other: `(location, int|float)` picks a wrong one in PySide2
    (the three planes then read one texture), `(location, QVector2D)` fails there, and the name-as-str forms want bytes in
    PySide6. `setUniformValue1i`, `setUniformValue1f` and `(location, x, y)` exist in both.
    """
    location = program.uniformLocation(name)
    if len(values) == 2:
        program.setUniformValue(location, float(values[0]), float(values[1]))
    elif isinstance(values[0], int):
        program.setUniformValue1i(location, values[0])
    else:
        program.setUniformValue1f(location, float(values[0]))
