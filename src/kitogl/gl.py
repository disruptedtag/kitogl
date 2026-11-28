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

    The location is looked up once per program and kept on the program object (programs here are linked once and never
    again, so a location cannot change). `glGetUniformLocation` is a query: the driver answers it only after catching up
    with what is queued before it, and asked ~30 times per frame at 180 Hz it kept the interface thread busy for a whole
    core with an NVIDIA card (0.97 cores in KitoGL 0.1.0; 0.06-0.18 with the locations kept, 1080p and 4K video).
    """
    locations = getattr(program, "_kitogl_uniforms", None)
    if locations is None:
        locations = {}
        program._kitogl_uniforms = locations
    location = locations.get(name)
    if location is None:
        location = locations[name] = program.uniformLocation(name)
    if len(values) == 2:
        program.setUniformValue(location, float(values[0]), float(values[1]))
    elif isinstance(values[0], int):
        program.setUniformValue1i(location, values[0])
    else:
        program.setUniformValue1f(location, float(values[0]))
