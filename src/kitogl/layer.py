"""`GLVideoLayer`: a video picture and an RGBA overlay, drawn by the graphics card, with scene transitions.

It is a child of a host window (a native `QOpenGLWindow` on Windows, a `QOpenGLWidget` elsewhere) that covers it while a
video plays. The host hands it pictures (`set_picture_source`, see `kitogl.picture`) and an optional overlay
(`set_text`: a premultiplied RGBA image and the rectangle where it has ink).

    decoder (planes) -> picture source -> this layer: 3 textures + one shader -> the screen

What the shader does per output pixel:

- reads Y, U and V (three 8-bit textures, uploaded as the decoder left them) and makes RGB with the matrix and range
  the picture names (`kitogl.color`);
- "covers" the screen (the picture is centred and cropped, never stretched);
- when it shrinks, averages the source pixels under the output pixel (a box filter, close to OpenCV's `INTER_AREA`)
  with a few bilinear taps; when it enlarges it is plain bilinear.

The overlay is uploaded once per change and drawn on top with premultiplied blending: only its ink rectangle.

The GLSL has no `#version` and nothing newer than 1.20 / ES 1.00, so desktop OpenGL 2.1 and the OpenGL ES 2 that ANGLE
gives Windows 7 both take it. Textures are 8-bit one-channel (`R8`, or `LUMINANCE` where the card has no `R8`).

Anything that goes wrong (no context, a shader that does not compile, a software renderer) marks the layer broken and
emits `failed`; the host then stops using it and draws by other means (a CPU path).
"""

from __future__ import annotations

import logging
import time
import ctypes
import sys
from array import array
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from kitogl._qt import (
    QImage,
    QOpenGLBuffer,
    QOpenGLContext,
    QOpenGLFramebufferObject,
    QOpenGLFunctions_2_1,
    QOpenGLPixelTransferOptions,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLWidget,
    QOpenGLWindow,
    QRect,
    QSurfaceFormat,
    Qt,
    QTimer,
    Signal,
    gl_proc_address,
)
from kitogl.color import COEFFICIENTS
from kitogl.frame_updates import FrameUpdates
from kitogl.geometry import DIRECTIONS, cover_geometry, eased, taps_for
from kitogl.gl import is_software_renderer, set_uniform
from kitogl.shaders import RGBA as _RGBA, SCENE as _SCENE, VERTEX as _VERTEX, yuv_source
from kitogl.trace import trace, trace_memory

if TYPE_CHECKING:
    from kitogl.picture import Picture

_LOG = logging.getLogger(__name__)

GL_FLOAT, GL_TRIANGLE_STRIP, GL_BLEND = 0x1406, 0x0005, 0x0BE2
GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_TEXTURE0 = 1, 0x0303, 0, 0x84C0
GL_COLOR_BUFFER_BIT, GL_RENDERER = 0x4000, 0x1F01

# GL objects whose context is gone (`GLVideoLayer.initializeGL` says why they are kept, not destroyed).
_GRAVEYARD: list = []


def _program(vertex: str, fragment: str) -> QOpenGLShaderProgram:
    program = QOpenGLShaderProgram()
    if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex):
        raise RuntimeError("vertex shader: " + str(program.log()))
    if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment):
        raise RuntimeError("fragment shader: " + str(program.log()))
    program.bindAttributeLocation("corner", 0)
    if not program.link():
        raise RuntimeError("shader link: " + str(program.log()))
    return program


_NATIVE_SURFACE = sys.platform == "win32"
_Surface = QOpenGLWindow if _NATIVE_SURFACE else QOpenGLWidget


class GLVideoLayer(_Surface):
    presentation_tick = Signal(float, float)
    scene_presented = Signal()
    failed = Signal()  # it cannot draw: whoever owns it goes back to the CPU path

    def __init__(self, parent=None, allow_software_renderer: bool = False, cuda_uploader_factory=None) -> None:
        """`allow_software_renderer`: draw even on a software OpenGL (llvmpipe, "GDI Generic"), for tests.
        `cuda_uploader_factory`: makes the object that copies a CUDA-resident picture (`cuda_resident = True`) into the
        layer's textures: `upload(picture, textures)` and `close()`, called with the GL context current. Without one,
        such a picture is refused and the layer fails."""
        if _NATIVE_SURFACE:
            super().__init__(QOpenGLWindow.NoPartialUpdate)
            self.setFlags(Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput)
            if parent is not None:
                parent.winId()
                self.setParent(parent.windowHandle())
        else:
            super().__init__(parent)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(Qt.WA_OpaquePaintEvent, True)
            self.setAttribute(Qt.WA_ShowWithoutActivating, True)
            self.setUpdateBehavior(QOpenGLWidget.NoPartialUpdate)
        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setSwapInterval(1)
        self.setFormat(fmt)
        self.allow_software_renderer = bool(allow_software_renderer)
        self._cuda_uploader_factory = cuda_uploader_factory
        self._static_destination = False
        self._waiting_for_picture = False
        self._hold_picture = False
        self._source: Optional[Callable[[], Optional[Picture]]] = None
        self._broken = False
        self._ready = False
        self._gl = None
        self._yuv: dict = {}  # taps per axis -> the shader program for it (made when first needed)
        self._outgoing_source = None
        self._outgoing_fbo = None
        self._outgoing_state = None
        self._scene_program = None
        self._scene_fbo = None
        self._scene_texture = None
        self._scene_source = None
        self._scene_snapshot = False
        self._snapshot_video = False
        self._snapshot_offset = (0.0, 0.0)
        self._capture_fbo = self._capture_target = None
        self._snapshot_fbo = None
        self._capturing_overlay = False
        self._scene_uploaded = None
        self._scene_clock = None
        self._text_only = False
        self._scene_spec = None
        self._rgba = None
        self._vbo = None
        self._cuda_upload = None
        self._textures: list = []
        self._texture_shapes: tuple = ()
        self._uploaded: Optional[Picture] = None
        self._r8 = True
        self._options = None
        self._text_source: Optional[QImage] = None  # the layer the window gave (identity decides whether to re-upload)
        self._text_ink = QRect()
        self._text_texture = None
        self._text_dirty = False
        self.renderer_name = ""
        self.frames_drawn = 0
        self._present_continuously = False
        self._clock_capable = False
        self._presentation_rate = 60.0
        self._output_fps = 0
        self._refresh_hz = 60.0
        self._swap_interval = 1
        self._vsync = True
        self._swap_control = None
        self._display_key = None
        self._last_swap = None
        self._timer_rate = 0.0
        self._refresh_epoch = None
        self._refresh_count = 0
        self._refresh_steps = 0
        self._fast_swaps = 0
        self._frame_updates = FrameUpdates(self, self.update)
        self._notify_scene_swap = False
        self._scene_drawn = False
        self.frameSwapped.connect(self._notify_scene_presented)
        if _NATIVE_SURFACE:
            self.frameSwapped.connect(self._present_next)

    def _pixel_ratio(self) -> float:
        return self.devicePixelRatio() if _NATIVE_SURFACE else self.devicePixelRatioF()

    def set_presentation_rate(self, rate: float) -> None:
        self._presentation_rate = max(1.0, float(rate))
        self._configure_presentation()

    def set_output_fps(self, fps: int) -> None:
        self._output_fps = fps
        if self._timer_rate:
            self._pace_with_timer()
        else:
            self._apply_frame_rate()

    def set_vsync(self, enabled: bool) -> None:
        if self._vsync == enabled:
            return
        self._vsync = enabled
        if not self._ready:
            surface_format = self.format()
            surface_format.setSwapInterval(int(enabled))
            self.setFormat(surface_format)
        self._timer_rate = 0.0
        self._frame_updates.reset()
        self._apply_frame_rate()
        self._present_continuously = self._clock_capable
        self._configure_presentation()
        self.request_animation_frame()

    def request_animation_frame(self) -> None:
        self._frame_updates.request()

    def _apply_frame_rate(self) -> None:
        self._frame_updates.set_rate(self._timer_rate or self._output_fps)

    def _pace_with_timer(self) -> None:
        # Use the source cadence in steady playback and the display rate during a mix.
        target = self._refresh_hz if self._scene_clock is not None else self._presentation_rate
        rate = min(self._output_fps or target, self._refresh_hz)
        if rate != self._timer_rate:
            self._timer_rate = rate
            self._refresh_epoch = None
            self._apply_frame_rate()

    def _configure_presentation(self) -> None:
        if not _NATIVE_SURFACE or not self._ready:
            return
        screen = self.screen()
        refresh = max(1.0, float(screen.refreshRate())) if screen is not None else 60.0
        key = (refresh, self._presentation_rate, self._vsync)
        if key == self._display_key:
            return
        # Present on physical refreshes. A multi-refresh driver wait could stall drawing and
        # then wait again in SwapBuffers; source cadence belongs to the display clock below.
        interval = int(self._vsync)
        if self._swap_control is None:
            interval = max(0, self.format().swapInterval())
        else:
            current = QOpenGLContext.currentContext() == self.context()
            if not current:
                self.makeCurrent()
            try:
                if not self._swap_control(interval):
                    interval = self._swap_interval
            finally:
                if not current:
                    self.doneCurrent()
        changed = (refresh, interval) != (self._refresh_hz, self._swap_interval)
        self._refresh_hz, self._swap_interval = refresh, interval
        self._display_key = key
        if changed:
            self._last_swap = None
            self._refresh_epoch = None
            self._refresh_count = 0
            self._refresh_steps = 0
        self._fast_swaps = 0

    def _display_refreshes(self, now: float) -> float:
        """Count elapsed refreshes cumulatively, including late or duplicate Qt callbacks."""
        rate = self._timer_rate or self._refresh_hz
        if self._refresh_epoch is None:
            self._refresh_epoch, self._refresh_count = now, 0
            return 1
        count = round((now - self._refresh_epoch) * rate)
        steps = max(0, count - self._refresh_count)
        self._refresh_count = count
        return steps * self._refresh_hz / rate

    def notify_on_scene_swap(self) -> None:
        self._notify_scene_swap = True

    def _notify_scene_presented(self) -> None:
        if self._scene_drawn:
            self._scene_drawn = False
            self._notify_scene_swap = False
            self.scene_presented.emit()

    def _present_next(self) -> None:
        self._configure_presentation()
        now = time.perf_counter()
        elapsed = None if self._last_swap is None else now-self._last_swap
        self._last_swap = now
        period = max(1, self._swap_interval) / self._refresh_hz
        self._refresh_steps = self._display_refreshes(now)
        if elapsed is not None:
            self._fast_swaps = self._fast_swaps+1 if elapsed < period*0.5 else 0
        moving = not self._static_destination and self._source is not None and self._source() is not None
        animate = not self._broken and self.isExposed() and (moving or self._scene_clock is not None)
        if not animate:
            self._frame_updates.reset()
            self._last_swap = None
            self._refresh_epoch = None
            self._refresh_count = 0
            return
        if self._present_continuously and self._fast_swaps >= 8:
            self._present_continuously = False  # the driver is ignoring the requested swap interval
        if self._clock_capable and (not self._present_continuously or self._swap_interval == 0):
            self._pace_with_timer()
            self.request_animation_frame()
        elif self._present_continuously:
            self.request_animation_frame()

    def grabFramebuffer(self) -> QImage:  # noqa: N802 - Qt-mandated name
        if not _NATIVE_SURFACE or not self.isValid():
            return super().grabFramebuffer()
        # NoPartialUpdate swaps away the back buffer. Explicit snapshots must draw before reading it;
        # normal presentation never downloads the framebuffer.
        self.makeCurrent()
        try:
            self._paint()
            return super().grabFramebuffer()
        finally:
            self.doneCurrent()

    # -- What the window tells it ------------------------------------------------------------------------------

    def set_picture_source(self, source: Optional[Callable[[], Optional[Picture]]]) -> None:
        self._source = source

    def set_waiting_for_picture(self, waiting: bool) -> None:
        self._waiting_for_picture = waiting

    def hold_current_picture(self) -> None:
        self._hold_picture = True

    def static_image(self):
        return self._text_source if self._static_destination else None

    @property
    def snapshot_video(self) -> bool:
        """True when the scene kept by `capture_scene` still has a live video in it."""
        return self._snapshot_video

    def current_text(self) -> Tuple[Optional[QImage], QRect]:
        """The text layer and the ink rectangle given by `set_text` (what an outgoing scene needs to keep drawing)."""
        return self._text_source, self._text_ink

    def set_static_destination(self, enabled: bool) -> None:
        if (self._static_destination and not enabled and not self._text_dirty
                and self._text_texture is not None and self._scene_texture is None
                and self._text_source is not None and self._scene_source is not None
                and self._text_source.cacheKey() == self._scene_source.cacheKey()):
            # The old static scene is already uploaded in the same pixel format and orientation.
            self._scene_texture, self._text_texture = self._text_texture, None
            self._scene_uploaded = self._scene_source
            self._text_dirty = True
        self._static_destination = enabled
        self._hold_picture = False

    def set_text(self, layer: Optional[QImage], ink: Optional[QRect]) -> None:
        """The text layer over the video (None = no text). Uploaded on the next paint, and only if it is a new one."""
        if layer is self._text_source and (ink or QRect()) == self._text_ink:
            return
        self._text_source = layer
        self._text_ink = QRect(ink) if ink is not None else QRect()
        self._text_dirty = True

    def set_scene_transition(self, source, spec=None, clock=None, text_only: bool = False, snapshot: bool = False) -> None:
        """Uploads the outgoing scene once; incoming video and text remain GPU-rendered each step."""
        self._scene_source, self._scene_spec, self._scene_clock = source, spec, clock
        self._notify_scene_swap = self._scene_drawn = False
        self._text_only = text_only
        self._scene_snapshot = snapshot

    def _bind_output(self) -> None:
        if self._capture_target is not None:
            self._capture_target.bind()
        else:
            QOpenGLFramebufferObject.bindDefault()

    def capture_scene(self) -> bool:
        """Retain static contributions while the outgoing or newly uncovered video stays live."""
        if self._broken or not self._ready or not self.isValid():
            return False
        current = QOpenGLContext.currentContext() == self.context()
        if not current:
            self.makeCurrent()
        texture = None
        clock = self._scene_clock
        amount = clock() if clock is not None else 1.0
        outgoing_live = self._outgoing_source is not None and self._outgoing_source() is not None
        incoming_live = (not outgoing_live and not self._static_destination and self._source is not None
                         and self._source() is not None)
        live = outgoing_live or incoming_live
        offset = self._snapshot_offset if self._scene_snapshot and self._snapshot_video else (0.0, 0.0)
        if live and self._scene_spec.kind == "push":
            dx, dy = DIRECTIONS[self._scene_spec.direction]
            displacement = eased(amount)-1.0 if incoming_live else eased(amount)
            offset = (offset[0]+dx*displacement, offset[1]+dy*displacement)
        try:
            ratio = self._pixel_ratio()
            width, height = max(1, round(self.width()*ratio)), max(1, round(self.height()*ratio))
            if self._capture_fbo is None or self._capture_fbo.width() != width or self._capture_fbo.height() != height:
                self._capture_fbo = QOpenGLFramebufferObject(width, height)
                if not self._capture_fbo.isValid():
                    raise RuntimeError("interrupted scene framebuffer unavailable")
            self._capture_target = self._capture_fbo
            self._capturing_overlay = live
            self._scene_clock = lambda: amount
            self._capture_target.bind()
            self._paint()
            texture = QOpenGLTexture(QOpenGLTexture.Target2D)
            texture.setSize(width, height)
            texture.setFormat(QOpenGLTexture.RGBA8_UNorm)
            texture.allocateStorage()
            texture.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
            texture.setWrapMode(QOpenGLTexture.ClampToEdge)
            texture.bind()
            self._gl.glCopyTexSubImage2D(0x0DE1, 0, 0, 0, 0, 0, width, height)
            if self._scene_texture is not None:
                self._scene_texture.destroy()
            self._scene_texture = texture
            self._scene_source = self._scene_uploaded = None
            self._scene_snapshot = True
            self._snapshot_video = live
            self._snapshot_offset = offset if live else (0.0, 0.0)
            return True
        except Exception:
            if texture is not None:
                texture.destroy()
            _LOG.warning("could not retain the interrupted GPU scene", exc_info=True)
            return False
        finally:
            self._scene_clock = clock
            self._capture_target = None
            self._capturing_overlay = False
            self._bind_output()
            if not current:
                self.doneCurrent()

    def _draw_rgba_texture(self, texture: int, offset=(0.0, 0.0), blend: bool = True) -> None:
        program = self._rgba
        program.bind()
        gl = self._gl
        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(0x0DE1, texture)
        set_uniform(program, "tex", 0)
        set_uniform(program, "box_a", -1.0, -1.0)
        set_uniform(program, "box_b", 1.0, 1.0)
        dx, dy = offset
        set_uniform(program, "uv_a", -dx, 1.0+dy)
        set_uniform(program, "uv_b", 1.0-dx, dy)
        if blend:
            gl.glEnable(GL_BLEND)
            gl.glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        else:
            gl.glDisable(GL_BLEND)
        self._draw_quad(program)
        gl.glDisable(GL_BLEND)

    def set_outgoing_video(self, source, text) -> None:
        if self._outgoing_state is None and source is not None:
            empty = dict(_cuda_upload=None, _textures=[], _texture_shapes=(), _uploaded=None,
                         _text_source=None, _text_ink=QRect(), _text_texture=None, _text_dirty=True)
            if source == self._source and not self._text_only and self._textures:
                self._outgoing_state = {name: getattr(self, name) for name in empty}
                for name, value in empty.items():
                    setattr(self, name, value)
            else:
                self._outgoing_state = empty
        self._outgoing_source = source
        if self._outgoing_state is not None and text is not None:
            image, ink = text
            if image is not self._outgoing_state["_text_source"]:
                self._outgoing_state.update(_text_source=image, _text_ink=QRect(ink), _text_dirty=True)
        elif self._outgoing_state is not None and source is not None:
            self._outgoing_state.update(_text_source=None, _text_ink=QRect(), _text_dirty=True)

    def _clear_outgoing(self, release_buffer: bool = False) -> None:
        if self._outgoing_state is not None:
            if self._outgoing_state["_cuda_upload"] is not None:
                self._outgoing_state["_cuda_upload"].close()
            for texture in self._outgoing_state["_textures"] + ([self._outgoing_state["_text_texture"]]
                          if self._outgoing_state["_text_texture"] is not None else []):
                texture.destroy()
        self._outgoing_state = None
        if release_buffer:
            self._outgoing_fbo = None

    def _draw_outgoing(self, width: int, height: int, shared_picture=None) -> bool:
        picture = (shared_picture if self._outgoing_source is self._source else
                   (None if self._outgoing_source is None else self._outgoing_source()))
        if picture is None:
            return False
        if self._outgoing_fbo is None or self._outgoing_fbo.width() != width or self._outgoing_fbo.height() != height:
            self._outgoing_fbo = QOpenGLFramebufferObject(width, height)
            if not self._outgoing_fbo.isValid():
                raise RuntimeError("outgoing transition framebuffer unavailable")
        state = self._outgoing_state
        incoming = {name: getattr(self, name) for name in state}
        self._outgoing_fbo.bind()
        try:
            for name, value in state.items():
                setattr(self, name, value)
            if self._text_only or self._capturing_overlay:
                self._gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                self._gl.glClear(GL_COLOR_BUFFER_BIT)
            else:
                self._upload(picture)
                self._draw_video(picture, width, height)
            if self._scene_snapshot and self._snapshot_video:
                if not self._capturing_overlay and self._snapshot_offset != (0.0, 0.0):
                    buffer = self._snapshot_fbo
                    if buffer is None or buffer.width() != width or buffer.height() != height:
                        buffer = QOpenGLFramebufferObject(width, height)
                        if not buffer.isValid():
                            raise RuntimeError("shifted snapshot framebuffer unavailable")
                        self._snapshot_fbo = buffer
                    buffer.bind()
                    self._draw_rgba_texture(self._outgoing_fbo.texture(), self._snapshot_offset, blend=False)
                self._draw_rgba_texture(self._scene_texture.textureId())
            else:
                self._draw_text()
        finally:
            for name in state:
                state[name] = getattr(self, name)
                setattr(self, name, incoming[name])
            self._bind_output()
        return True

    def _clear_scene(self, release_buffers: bool = False) -> None:
        self._clear_outgoing(release_buffers)
        if self._scene_texture is not None:
            self._scene_texture.destroy()
        self._scene_texture = self._scene_uploaded = None
        self._scene_snapshot = False
        self._snapshot_video = False
        self._snapshot_offset = (0.0, 0.0)
        self._capture_fbo = None
        self._snapshot_fbo = None
        if release_buffers:
            self._scene_fbo = None

    @property
    def broken(self) -> bool:
        return self._broken

    # -- OpenGL -----------------------------------------------------------------------------------------------

    def initializeGL(self) -> None:  # noqa: N802 - Qt-mandated name
        # Qt calls this AGAIN when it gives the widget a new context (the window moved to another screen or card, the driver
        # was reset, the layer was re-parented). Every GL object of the old context is gone, and what this object still
        # holds of them would draw black (a picture "already uploaded" that is not on the card any more). They are only
        # forgotten here, kept alive in `_GRAVEYARD`: a QOpenGLTexture destroyed while another context is current crashes
        # (found by `test_a_new_context_does_not_leave_the_layer_black`), and the card frees the old context's textures
        # itself when that context goes. It happens when a window changes screen or card: a few small objects, rarely.
        trace_memory(f"GPU layer: GL context initialised (graveyard {len(_GRAVEYARD)} objects), memory")
        self._ready = False
        self._close_cuda()
        if self._outgoing_state is not None:
            if self._outgoing_state["_cuda_upload"] is not None:
                self._outgoing_state["_cuda_upload"].close()
            _GRAVEYARD.extend(self._outgoing_state["_textures"] + [obj for obj in
                              (self._outgoing_state["_text_texture"], self._outgoing_fbo) if obj is not None])
            self._outgoing_state = self._outgoing_fbo = None
            self.set_outgoing_video(self._outgoing_source, None)
        if self._outgoing_fbo is not None:
            _GRAVEYARD.append(self._outgoing_fbo)
            self._outgoing_fbo = None
        _GRAVEYARD.extend(obj for obj in (
            self._scene_texture, self._scene_fbo, self._scene_program, self._capture_fbo, self._snapshot_fbo
        ) if obj is not None)
        self._scene_texture = self._scene_uploaded = self._scene_fbo = self._scene_program = None
        self._capture_fbo = self._capture_target = None
        self._snapshot_fbo = None
        self._scene_snapshot = self._snapshot_video = False
        self._snapshot_offset = (0.0, 0.0)
        _GRAVEYARD.extend(self._textures + list(self._yuv.values()) +
                          [obj for obj in (self._text_texture, self._rgba, self._vbo) if obj is not None])
        self._textures, self._texture_shapes, self._uploaded = [], (), None
        self._text_texture = None
        self._text_dirty = self._text_source is not None
        self._yuv = {}
        try:
            gl = QOpenGLFunctions_2_1()
            gl.initializeOpenGLFunctions()
            name = gl.glGetString(GL_RENDERER)
            self.renderer_name = name.decode("latin-1", "replace") if isinstance(name, (bytes, bytearray)) else str(name or "")
            if not self.allow_software_renderer and is_software_renderer(self.renderer_name):
                raise RuntimeError(f"software renderer ({self.renderer_name})")
            self._present_continuously = _NATIVE_SURFACE and not is_software_renderer(self.renderer_name)
            self._clock_capable = self._present_continuously
            context = self.context()
            version = context.format()
            extensions = {bytes(item).decode("latin-1") for item in context.extensions()}
            self._r8 = (version.majorVersion(), version.minorVersion()) >= (3, 0) or bool(
                {"GL_ARB_texture_rg", "GL_EXT_texture_rg"} & extensions
            )
            self._yuv_program(1)  # a shader the driver refuses is found now, not at the first picture
            self._rgba = _program(_VERTEX, _RGBA)
            self._scene_program = _program(_VERTEX, _SCENE)  # never link a shader during a live transition
            corners = array("f", (0, 0, 1, 0, 0, 1, 1, 1))  # C floats: 32 bits, what the vertex shader reads
            self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._vbo.create()
            self._vbo.bind()
            self._vbo.allocate(corners.tobytes(), len(corners) * corners.itemsize)
            self._options = QOpenGLPixelTransferOptions()
            self._options.setAlignment(1)
            # ctypes releases the GIL during the driver transfer. Qt's setData wrapper can hold
            # it long enough to starve both Python pacers while two large software frames upload.
            convention = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
            upload_type = convention(None, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)
            self._plane_upload = upload_type(gl_proc_address(self.context(), "glTexSubImage2D"))
            self._gl = gl
            self._ready = True
            self._display_key = None
            self._swap_control = None
            if _NATIVE_SURFACE and not context.isOpenGLES():
                try:
                    address = gl_proc_address(context, "wglSwapIntervalEXT")
                    self._swap_control = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(address)
                except (AttributeError, OSError):
                    pass
            self._configure_presentation()
            _LOG.info("GPU video layer ready on %s (%s textures)", self.renderer_name, "R8" if self._r8 else "LUMINANCE")
            trace("GPU layer: ready on %s (%s textures)" % (self.renderer_name, "R8" if self._r8 else "LUMINANCE"))
        except Exception as error:  # noqa: BLE001
            self._fail(error)

    def _yuv_program(self, taps: int) -> QOpenGLShaderProgram:
        program = self._yuv.get(taps)
        if program is None:
            program = self._yuv[taps] = _program(_VERTEX, yuv_source(taps))
        return program

    def paintGL(self) -> None:  # noqa: N802 - Qt-mandated name
        if self._broken or not self._ready:
            return
        try:
            if _NATIVE_SURFACE:
                self._configure_presentation()
                self.presentation_tick.emit(self._refresh_steps, self._refresh_hz)
                self._refresh_steps = 0
            before = self.frames_drawn
            self._paint()
            if self._notify_scene_swap:
                self._scene_drawn = self._scene_clock is not None and self.frames_drawn != before
        except Exception as error:  # noqa: BLE001
            self._fail(error)

    def _fail(self, error: BaseException) -> None:
        if self._broken:
            return
        self._broken = True
        _LOG.warning("GPU video layer disabled, the CPU path takes over: %s", error)
        QTimer.singleShot(0, self.failed.emit)

    # -- Drawing ----------------------------------------------------------------------------------------------

    def _paint(self) -> None:
        gl = self._gl
        ratio = self._pixel_ratio()
        width, height = max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))
        for name in ("_outgoing_fbo", "_scene_fbo"):
            buffer = getattr(self, name)
            if buffer is None or buffer.width() != width or buffer.height() != height:
                buffer = QOpenGLFramebufferObject(width, height)
                if not buffer.isValid():
                    raise RuntimeError("transition framebuffer unavailable")
                setattr(self, name, buffer)
        gl.glViewport(0, 0, width, height)
        gl.glDisable(GL_BLEND)
        picture = None
        if not self._static_destination:
            if not self._waiting_for_picture and self._source is not None:
                picture = self._source()
            if picture is None and self._hold_picture:
                picture = self._uploaded
        if self._outgoing_source is None and self._outgoing_state is not None:
            self._clear_outgoing()
        frozen_scene = self._scene_snapshot and not self._snapshot_video
        moving = not frozen_scene and self._outgoing_source is not None and self._draw_outgoing(width, height, picture)
        static_ready = self._static_destination and self._text_source is not None
        if static_ready and self._uploaded is not None:
            self._close_cuda()
            for texture in self._textures:
                texture.destroy()
            self._textures, self._texture_shapes, self._uploaded = [], (), None
        scene_ready = self._scene_source is not None or (self._scene_snapshot and self._scene_texture is not None) or moving
        if picture is None and not static_ready and scene_ready:
            self._draw_scene(amount=0.0, moving=moving)  # incoming decoder still opening: hold the full outgoing scene
            return
        if picture is None and not static_ready:
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
            return
        if picture is not None:
            self._upload(picture)
        transitioning = self._scene_clock is not None and scene_ready
        if transitioning:
            if self._scene_fbo is None or self._scene_fbo.width() != width or self._scene_fbo.height() != height:
                self._scene_fbo = QOpenGLFramebufferObject(width, height)
                if not self._scene_fbo.isValid():
                    raise RuntimeError("transition framebuffer unavailable")
            self._scene_fbo.bind()
        elif self._scene_texture is not None or self._scene_fbo is not None:
            self._clear_scene()
        if transitioning and self._text_only:
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
        elif picture is not None and not self._capturing_overlay:
            self._draw_video(picture, width, height)
        else:
            gl.glClearColor(0.0, 0.0, 0.0, 0.0 if self._capturing_overlay else 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
        self._draw_text(static=self._static_destination)
        if transitioning:
            self._bind_output()
            if self._text_only and not self._capturing_overlay:
                self._draw_video(picture, width, height)
            self._draw_scene(moving=moving)
        self.frames_drawn += 1
        if self.frames_drawn == 1:
            trace("GPU layer: first picture drawn")

    def _draw_scene(self, amount=None, moving=False) -> None:
        if self._scene_program is None:
            self._scene_program = _program(_VERTEX, _SCENE)
        if self._scene_source is not None and self._scene_uploaded is not self._scene_source:
            if self._scene_texture is not None:
                self._scene_texture.destroy()
            piece = self._scene_source.convertToFormat(QImage.Format_RGBA8888_Premultiplied)
            texture = QOpenGLTexture(QOpenGLTexture.Target2D)
            texture.setSize(piece.width(), piece.height())
            texture.setFormat(QOpenGLTexture.RGBA8_UNorm)
            texture.allocateStorage()
            texture.setData(QOpenGLTexture.RGBA, QOpenGLTexture.UInt8, bytes(piece.constBits()), self._options)
            texture.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
            texture.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._scene_texture, self._scene_uploaded = texture, self._scene_source
        program = self._scene_program
        program.bind()
        if self._text_only and not self._capturing_overlay:
            self._gl.glEnable(GL_BLEND)
            self._gl.glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)
        else:
            self._gl.glDisable(GL_BLEND)
        self._gl.glActiveTexture(GL_TEXTURE0)
        outgoing_buffer = (self._snapshot_fbo if self._snapshot_video and not self._capturing_overlay
                           and self._snapshot_offset != (0.0, 0.0) else self._outgoing_fbo)
        if moving:
            self._gl.glBindTexture(0x0DE1, outgoing_buffer.texture())
        else:
            self._scene_texture.bind()
        set_uniform(program, "moving_source", int(moving or self._scene_snapshot))
        self._gl.glActiveTexture(GL_TEXTURE0 + 1)
        fallback = outgoing_buffer.texture() if moving else self._scene_texture.textureId()
        incoming = self._scene_fbo.texture() if self._scene_fbo is not None else fallback
        self._gl.glBindTexture(0x0DE1, incoming)
        set_uniform(program, "outgoing", 0)
        set_uniform(program, "incoming", 1)
        set_uniform(program, "amount", float(self._scene_clock() if amount is None else amount))
        set_uniform(program, "softness", float(self._scene_spec.softness))
        set_uniform(program, "effect", {"fade": 0, "wipe": 1, "push": 2}.get(self._scene_spec.kind, 0))
        set_uniform(program, "direction", *DIRECTIONS[self._scene_spec.direction])
        set_uniform(program, "box_a", -1.0, -1.0)
        set_uniform(program, "box_b", 1.0, 1.0)
        set_uniform(program, "uv_a", 0.0, 0.0)
        set_uniform(program, "uv_b", 1.0, 1.0)
        self._draw_quad(program)
        self._gl.glDisable(GL_BLEND)

    def _new_texture(self, columns: int, rows: int, channels: int = 1):
        texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        texture.setSize(columns, rows)
        texture.setFormat(QOpenGLTexture.RG8_UNorm if channels == 2 else
                          (QOpenGLTexture.R8_UNorm if self._r8 else QOpenGLTexture.LuminanceFormat))
        texture.allocateStorage()
        texture.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
        texture.setWrapMode(QOpenGLTexture.ClampToEdge)
        return texture

    def _upload(self, picture: Picture) -> None:
        if picture is self._uploaded:
            return
        cuda = bool(getattr(picture, "cuda_resident", False))
        if cuda and not self._r8:
            raise RuntimeError("CUDA interop requires desktop GL with RG textures")
        shapes = tuple(picture.line_sizes)
        if shapes != self._texture_shapes:
            self._close_cuda()
            for texture in self._textures:
                texture.destroy()
            self._textures = [self._new_texture(columns, rows, 2 if cuda and index == 1 else 1)
                              for index, (columns, rows) in enumerate(picture.line_sizes)]
            self._texture_shapes = shapes
        # The chroma planes are half the size: swscale (the CPU path) repeats each chroma sample over its 2x2 pixels, so an edge
        # between two flat colours stays 2 pixels sharp. Interpolating them (linear) smeared it over 4 and drew a thin line.
        for texture in self._textures[1:]:
            texture.setMinMagFilters(QOpenGLTexture.Nearest, QOpenGLTexture.Nearest)
        if cuda:
            if self._cuda_upload is None:
                if self._cuda_uploader_factory is None:
                    raise RuntimeError("a CUDA-resident picture needs a cuda_uploader_factory")
                self._cuda_upload = self._cuda_uploader_factory()
            self._cuda_upload.upload(picture, self._textures)
        else:
            self._close_cuda()
            self._gl.glPixelStorei(0x0CF5, 1)  # GL_UNPACK_ALIGNMENT: PyAV rows include their own padding
            for texture, plane, (columns, rows) in zip(self._textures, picture.planes, picture.line_sizes):
                texture.bind()
                self._plane_upload(0x0DE1, 0, 0, 0, columns, rows,
                                   0x1903 if self._r8 else 0x1909,
                                   0x1401, plane.buffer_ptr)
        self._uploaded = picture

    def _draw_video(self, picture: Picture, width: int, height: int) -> None:
        gl = self._gl
        u0, v0, u1, v1, ratio = cover_geometry(picture.width, picture.height, width, height)
        program = self._yuv_program(taps_for(ratio))
        program.bind()
        set_uniform(program, "nv12", int(bool(getattr(picture, "cuda_resident", False))))
        for index, texture in enumerate(self._textures):
            gl.glActiveTexture(GL_TEXTURE0 + index)
            texture.bind()
            set_uniform(program, ("ty", "tu", "tv")[index], index)
        (stride_y, rows_y), (stride_c, rows_c) = picture.line_sizes[0], picture.line_sizes[1]
        chroma_w, chroma_h = ((picture.width + 1) // 2, (picture.height + 1) // 2)
        set_uniform(program, "box_a", -1.0, -1.0)
        set_uniform(program, "box_b", 1.0, 1.0)
        set_uniform(program, "uv_a", u0, v0)
        set_uniform(program, "uv_b", u1, v1)
        set_uniform(program, "scale_y", picture.width / stride_y, picture.height / rows_y)
        set_uniform(program, "scale_c", chroma_w / stride_c, chroma_h / rows_c)
        set_uniform(program, "limit_y", (picture.width - 0.5) / stride_y, (picture.height - 0.5) / rows_y)
        set_uniform(program, "limit_c", (chroma_w - 0.5) / stride_c, (chroma_h - 0.5) / rows_c)
        set_uniform(program, "footprint", ratio / picture.width, ratio / picture.height)
        red_v, green_u, green_v, blue_u = COEFFICIENTS[picture.matrix]
        set_uniform(program, "k_a", red_v, green_u)
        set_uniform(program, "k_b", green_v, blue_u)
        if picture.full_range:
            set_uniform(program, "range_a", 0.0, 1.0)
            set_uniform(program, "range_b", 1.0, 0.0)
        else:
            set_uniform(program, "range_a", 16.0 / 255.0, 255.0 / 219.0)
            set_uniform(program, "range_b", 255.0 / 224.0, 0.0)
        self._draw_quad(program)

    def _draw_quad(self, program) -> None:
        self._vbo.bind()
        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, GL_FLOAT, 0, 2, 8)
        self._gl.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def _draw_text(self, static: bool = False) -> None:
        gl = self._gl
        if self._text_dirty:
            self._text_dirty = False
            if self._text_texture is not None:
                self._text_texture.destroy()
                self._text_texture = None
            if self._text_source is not None and not self._text_ink.isEmpty():
                source = (self._text_source if static or self._text_source.size() == self._text_ink.size()
                          else self._text_source.copy(self._text_ink))
                piece = source.convertToFormat(QImage.Format_RGBA8888_Premultiplied)
                texture = QOpenGLTexture(QOpenGLTexture.Target2D)
                texture.setSize(piece.width(), piece.height())
                texture.setFormat(QOpenGLTexture.RGBA8_UNorm)
                texture.allocateStorage()
                texture.setData(QOpenGLTexture.RGBA, QOpenGLTexture.UInt8, bytes(piece.constBits())[: piece.width() * piece.height() * 4], self._options)
                texture.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
                texture.setWrapMode(QOpenGLTexture.ClampToEdge)
                self._text_texture = texture
        if self._text_texture is None or self._text_source is None:
            return
        ratio = self._pixel_ratio()
        layer_w, layer_h = max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))
        ink = QRect(0, 0, layer_w, layer_h) if static else self._text_ink
        program = self._rgba
        program.bind()
        gl.glActiveTexture(GL_TEXTURE0)
        self._text_texture.bind()
        set_uniform(program, "tex", 0)
        set_uniform(program, "box_a", ink.x() / layer_w * 2.0 - 1.0, 1.0 - (ink.y() + ink.height()) / layer_h * 2.0)
        set_uniform(program, "box_b", (ink.x() + ink.width()) / layer_w * 2.0 - 1.0, 1.0 - ink.y() / layer_h * 2.0)
        set_uniform(program, "uv_a", 0.0, 0.0)
        set_uniform(program, "uv_b", 1.0, 1.0)
        gl.glEnable(GL_BLEND)
        gl.glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)  # retain shadow alpha on transparent scene buffers
        self._draw_quad(program)
        gl.glDisable(GL_BLEND)

    def _close_cuda(self) -> None:
        uploader, self._cuda_upload = self._cuda_upload, None
        if uploader is not None:
            uploader.close()

    def _free_gl_objects(self) -> None:
        self._close_cuda()
        self._clear_scene(release_buffers=True)
        self._scene_program = None
        self._rgba = None  # destroy its program while this context is current, not during a later GC
        for texture in self._textures + ([self._text_texture] if self._text_texture is not None else []):
            texture.destroy()
        self._yuv = {}
        if self._vbo is not None:
            self._vbo.destroy()

    # -- Giving the card's memory back ------------------------------------------------------------------------

    def trim(self) -> None:
        """Gives back the textures and shaders but keeps the widget and its context (the next picture makes them again)."""
        if not self._ready or not self.isValid():
            return
        self._frame_updates.reset()
        self.makeCurrent()
        try:
            self._close_cuda()
            self._clear_scene(release_buffers=True)
            self._scene_program = None
            for texture in self._textures + ([self._text_texture] if self._text_texture is not None else []):
                texture.destroy()
            self._yuv = {}
            self._textures, self._texture_shapes, self._uploaded = [], (), None
            self._text_texture = None
            self._text_dirty = self._text_source is not None
        finally:
            self.doneCurrent()

    def release(self) -> None:
        """Destroys the textures and shaders (with the context current, as OpenGL wants). Call before deleting."""
        if not self._ready or not self.isValid():
            return
        self._frame_updates.reset()
        self.makeCurrent()
        try:
            self._free_gl_objects()
        finally:
            self._textures, self._text_texture, self._vbo, self._yuv = [], None, None, {}
            self._texture_shapes, self._uploaded, self._ready = (), None, False
            self._scene_source = self._scene_clock = self._scene_spec = None
            self.doneCurrent()
