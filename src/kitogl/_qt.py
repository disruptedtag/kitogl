"""The Qt binding KitoGL draws with, and the few names that live in different modules in Qt 5 and Qt 6.

Which binding: the one the application already imported (KitoGL must never load a second Qt into a process), else the
`KITOGL_QT` environment variable ("pyside6" or "pyside2"), else PySide6 if it is installed, else PySide2.

The OpenGL helper classes live in `QtGui` in Qt 5 (PySide2) and in `QtOpenGL` in Qt 6 (PySide6); `QOpenGLWidget` lives in
`QtWidgets` in Qt 5 and in `QtOpenGLWidgets` in Qt 6. Everything KitoGL needs is imported from here.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def _choose() -> str:
    for name in ("PySide6", "PySide2"):
        if name in sys.modules:
            return name
    wanted = os.environ.get("KITOGL_QT", "").strip().lower()
    if wanted in ("pyside6", "pyside2"):
        name = "PySide6" if wanted == "pyside6" else "PySide2"
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"KITOGL_QT={wanted!r} but {name} is not installed")
        return name
    for name in ("PySide6", "PySide2"):
        if importlib.util.find_spec(name) is not None:
            return name
    raise ImportError("KitoGL needs PySide6 or PySide2")


QT_API: str = _choose()
IS_QT6: bool = QT_API == "PySide6"

_core = importlib.import_module(QT_API + ".QtCore")
_gui = importlib.import_module(QT_API + ".QtGui")
_widgets = importlib.import_module(QT_API + ".QtWidgets")

QObject = _core.QObject
QRect = _core.QRect
Qt = _core.Qt
QTimer = _core.QTimer
Signal = _core.Signal

QGuiApplication = _gui.QGuiApplication
QImage = _gui.QImage
QOffscreenSurface = _gui.QOffscreenSurface
QOpenGLContext = _gui.QOpenGLContext
QSurfaceFormat = _gui.QSurfaceFormat

if IS_QT6:
    _opengl = importlib.import_module("PySide6.QtOpenGL")
    QOpenGLWidget = importlib.import_module("PySide6.QtOpenGLWidgets").QOpenGLWidget
    QOpenGLFunctions_2_1 = _opengl.QOpenGLFunctions_2_1
else:
    _opengl = _gui
    QOpenGLWidget = _widgets.QOpenGLWidget
    # Qt 5 has one `QOpenGLFunctions` with everything the compositor calls (draw, blend, texture units, clear);
    # Qt 6 wraps it per version in `QOpenGLFunctions_2_1`. Same methods, same `initializeOpenGLFunctions()`.
    QOpenGLFunctions_2_1 = _gui.QOpenGLFunctions

QOpenGLBuffer = _opengl.QOpenGLBuffer
QOpenGLFramebufferObject = _opengl.QOpenGLFramebufferObject
QOpenGLPixelTransferOptions = _opengl.QOpenGLPixelTransferOptions
QOpenGLShader = _opengl.QOpenGLShader
QOpenGLShaderProgram = _opengl.QOpenGLShaderProgram
QOpenGLTexture = _opengl.QOpenGLTexture
QOpenGLWindow = _opengl.QOpenGLWindow

_gl_libraries: dict = {}


def gl_proc_address(context, name: str) -> int:
    """Core GL entry points; PySide2 omits QOpenGLContext.getProcAddress from its bindings."""
    getter = getattr(context, "getProcAddress", None)
    if getter is not None:
        address = getter(name)
        if address:
            return int(address)
    import ctypes
    import ctypes.util

    library_name = "GLESv2" if context.isOpenGLES() else ("OpenGL" if sys.platform == "darwin" else "GL")
    library = _gl_libraries.get(library_name)
    if library is None:
        if sys.platform == "win32":
            library = ctypes.WinDLL("libGLESv2.dll" if context.isOpenGLES() else "opengl32.dll")
        else:
            library = ctypes.CDLL(ctypes.util.find_library(library_name))
        _gl_libraries[library_name] = library
    if sys.platform == "win32" and not context.isOpenGLES():
        library.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        library.wglGetProcAddress.restype = ctypes.c_void_p
        address = library.wglGetProcAddress(name.encode("ascii"))
        if address not in (None, 0, 1, 2, 3, ctypes.c_void_p(-1).value):
            return int(address)
    return int(ctypes.cast(getattr(library, name), ctypes.c_void_p).value)
