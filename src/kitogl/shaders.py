"""The GLSL sources of the compositor. No Qt.

No `#version` and nothing newer than GLSL 1.20 / ES 1.00, so desktop OpenGL 2.1 and the OpenGL ES 2 that ANGLE gives
Windows 7 and 8 both take them. Textures are 8-bit (`R8`, or `LUMINANCE` where the card has no `R8`).

- `VERTEX` places one quad and maps picture coordinates onto it.
- `yuv_source(taps)` converts Y/U/V (or NV12) to RGB with a matrix and range, covers the screen and shrinks with a box
  filter of `taps` x `taps` bilinear samples, written out with no loop.
- `RGBA` draws a premultiplied RGBA texture (the overlay, a captured scene).
- `SCENE` mixes two scenes: fade, wipe (soft edge) or push, along a direction.
"""

from __future__ import annotations

VERTEX = """
attribute vec2 corner;
varying vec2 v_uv;
uniform vec2 box_a;   // NDC of the bottom-left corner
uniform vec2 box_b;   // NDC of the top-right corner
uniform vec2 uv_a;    // picture coordinates (0..1, v down) of the TOP-left corner
uniform vec2 uv_b;    // ... and of the BOTTOM-right one
void main() {
  gl_Position = vec4(mix(box_a, box_b, corner), 0.0, 1.0);
  v_uv = mix(uv_a, uv_b, vec2(corner.x, 1.0 - corner.y));
}
"""

YUV_HEAD = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec2 v_uv;
uniform sampler2D ty; uniform sampler2D tu; uniform sampler2D tv; uniform int nv12;
uniform vec2 scale_y;  uniform vec2 scale_c;   // picture coords -> texture coords (rows are padded)
uniform vec2 limit_y;  uniform vec2 limit_c;   // last texel centre: never read the padding
uniform vec2 footprint;                        // size of one output pixel in picture coords
uniform vec2 k_a; uniform vec2 k_b;            // (r from v, g from u) (g from v, b from u)
uniform vec2 range_a;                          // (luma offset, luma scale)
uniform vec2 range_b;                          // (chroma scale, unused)
vec3 yuv(vec2 uv) {
  float y = (texture2D(ty, min(uv * scale_y, limit_y)).r - range_a.x) * range_a.y;
  vec2 c = min(uv * scale_c, limit_c);
  float u = (texture2D(tu, c).r - 0.5) * range_b.x;
  float v = ((nv12 != 0 ? texture2D(tu, c).g : texture2D(tv, c).r) - 0.5) * range_b.x;
  return vec3(y + k_a.x * v, y - k_a.y * u - k_b.x * v, y + k_b.y * u);
}
"""


def yuv_source(taps: int) -> str:
    """The fragment shader for `taps` bilinear samples per axis, with no loop: the samples are written out.

    GLSL ES 1.00 (ANGLE on Windows 7/8) is strict about loops that contain texture reads and a `break` on a uniform,
    and a shader that does not compile there would make the whole path give up; straight-line code is also what the card
    runs fastest. The sum is the same as the loop it replaces, sample for sample.
    """
    taps = int(min(4, max(1, taps)))
    body = []
    for j in range(taps):
        for i in range(taps):
            offset_x, offset_y = (i + 0.5) / taps - 0.5, (j + 0.5) / taps - 0.5
            body.append(f"  sum += yuv(v_uv + vec2({offset_x:.6f}, {offset_y:.6f}) * footprint);")
    return (
        YUV_HEAD + "void main() {\n  vec3 sum = vec3(0.0);\n" + "\n".join(body)
        + f"\n  gl_FragColor = vec4(clamp(sum / {float(taps * taps):.1f}, 0.0, 1.0), 1.0);\n}}\n"
    )


RGBA = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec2 v_uv;
uniform sampler2D tex;
void main() { gl_FragColor = texture2D(tex, v_uv); }
"""


SCENE = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec2 v_uv;
uniform sampler2D outgoing; uniform sampler2D incoming;
uniform float amount; uniform float softness; uniform int effect; uniform int moving_source;
uniform vec2 direction;
vec4 old_pixel(vec2 uv) { return texture2D(outgoing, moving_source != 0 ? vec2(uv.x, 1.0 - uv.y) : uv); }
vec4 next_pixel(vec2 uv) { return texture2D(incoming, vec2(uv.x, 1.0 - uv.y)); }
void main() {
  float t = amount * amount * (3.0 - 2.0 * amount);
  vec4 a = old_pixel(v_uv);
  vec4 b = next_pixel(v_uv);
  if (effect == 2) {
    vec2 delta = vec2(direction.x, direction.y) * t;
    vec2 old_uv = v_uv - delta;
    vec2 new_uv = v_uv - delta + direction;
    if (old_uv.x >= 0.0 && old_uv.x <= 1.0 && old_uv.y >= 0.0 && old_uv.y <= 1.0)
      gl_FragColor = old_pixel(old_uv);
    else gl_FragColor = next_pixel(new_uv);
  } else if (effect == 1) {
    float coordinate = abs(direction.x) > 0.5 ? v_uv.x : v_uv.y;
    float sign_axis = direction.x + direction.y;
    float edge = (sign_axis > 0.0 ? t : 1.0 - t) * (1.0 + softness) - softness * 0.5;
    float mask = softness > 0.0 ? clamp((coordinate - edge) / softness + 0.5, 0.0, 1.0) : step(edge, coordinate);
    if (sign_axis > 0.0) mask = 1.0 - mask;
    if (amount <= 0.0) mask = 0.0;
    if (amount >= 1.0) mask = 1.0;
    gl_FragColor = mix(a, b, mask);
  } else gl_FragColor = mix(a, b, t);
}
"""
