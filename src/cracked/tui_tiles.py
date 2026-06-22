"""
Custom tile-face artwork for the terminal UI (higher-resolution).

Each tile is rendered as a local pixel bitmap (FW x FH of hex colours, None =
transparent), drawn on an ivory body with a dark border. Number suits use our
own pip artwork (bamboo sticks, circle rings); characters, winds and dragons use
the *real* CJK symbols, rasterised from the Unicode Mahjong glyphs (the inner
symbol is extracted and tinted by suit). If Pillow or the symbol font is
unavailable, those tiles fall back to a small Latin pixel font so the module
still works everywhere.

The faces are sampled in a tile's rotated local frame by the renderers, so the
artwork rotates with the tile.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

from rich.style import Style
from rich.text import Text

# ---------------------------------------------------------------------------
# Palette + dimensions
# ---------------------------------------------------------------------------

FELT = "#0a3024"
FW, FH = 32, 44                        # face bitmap size (pixels)

_IVORY = "#efe4c2"
_BORDER = "#6f5f33"
_BAMBOO = "#2e8b57"
_BAMBOO_NODE = "#1d5e3a"
_CIRCLE = "#2a78c8"
_CHAR = "#c0392b"
_GOLD = "#caa12a"
_DRAGON = {31: "#c0392b", 32: "#2ca05a", 33: "#3a6ea5"}   # red / green / white(frame)
_BONUS_COLOR = {
    34: "#2ca05a", 35: "#c0392b", 36: "#d07e1a", 37: "#3a6ea5",   # spring/summer/autumn/winter
    38: "#c0392b", 39: "#7d4fa0", 40: "#caa12a", 41: "#2e8b57",   # plum/orchid/chrysanth./bamboo
    42: "#9c6b3f", 43: "#6b6b73", 44: "#c0392b", 45: "#3a9a4a",   # cat/mouse/cockerel/worm
}

_SYM_W, _SYM_H = FW - 8, FH - 10       # area for a rasterised symbol
_FONT_PATH = "/usr/share/fonts/TTF/Symbola.ttf"

Face = list[list[Optional[str]]]

# pip slot grid (3x3) → face coordinates
_PX = [8, 16, 24]
_PY = [12, 22, 32]
_PIPS = {
    1: [(1, 1)],
    2: [(0, 0), (2, 2)],
    3: [(0, 0), (1, 1), (2, 2)],
    4: [(0, 0), (2, 0), (0, 2), (2, 2)],
    5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
    6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)],
    7: [(0, 0), (2, 0), (1, 1), (0, 1), (2, 1), (0, 2), (2, 2)],
    8: [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
    9: [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
}

# minimal 3x5 Latin fallback font (only used if rasterisation is unavailable)
_FALLBACK = {
    "1": ["010", "110", "010", "010", "111"], "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"], "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"], "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"], "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"], "E": ["111", "100", "110", "100", "111"],
    "S": ["111", "100", "111", "001", "111"], "W": ["101", "101", "101", "111", "101"],
    "N": ["101", "111", "111", "111", "101"], "R": ["110", "101", "110", "101", "101"],
    "G": ["111", "100", "101", "101", "111"],
}


# ---------------------------------------------------------------------------
# Low-level drawing
# ---------------------------------------------------------------------------

def _put(f: Face, x: int, y: int, c: str) -> None:
    if 0 <= x < FW and 0 <= y < FH:
        f[y][x] = c


def _blank_face() -> Face:
    f: Face = [[_IVORY] * FW for _ in range(FH)]
    for x in range(FW):
        f[0][x] = _BORDER
        f[FH - 1][x] = _BORDER
    for y in range(FH):
        f[y][0] = _BORDER
        f[y][FW - 1] = _BORDER
    r = 4                              # rounded corners
    for ax, ay, sx, sy in [(0, 0, 1, 1), (FW - 1, 0, -1, 1),
                           (0, FH - 1, 1, -1), (FW - 1, FH - 1, -1, -1)]:
        for i in range(r):
            for j in range(r):
                if i + j < r:
                    f[ay + sy * j][ax + sx * i] = None
    return f


def _ring(f: Face, cx: int, cy: int, color: str) -> None:
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            d = math.hypot(dx, dy)
            if 2.7 <= d <= 4.3:
                _put(f, cx + dx, cy + dy, color)


def _stick(f: Face, cx: int, cy: int, color: str, half: int = 6) -> None:
    for dy in range(-half, half):
        _put(f, cx, cy + dy, color)
        _put(f, cx + 1, cy + dy, color)
    for dx in (-1, 2):
        _put(f, cx + dx, cy, _BAMBOO_NODE)
        _put(f, cx + dx, cy - 1, _BAMBOO_NODE)


def _pips(f: Face, n: int, color: str, stick: bool) -> None:
    for col, row in _PIPS[n]:
        cx, cy = _PX[col], _PY[row]
        (_stick if stick else _ring)(f, cx, cy, color)


def _seg(f: Face, x0: int, y0: int, x1: int, y1: int, color: str, node: bool = True) -> None:
    """A bamboo stick drawn as a 2px segment between two points (may be slanted)."""
    n = max(abs(x1 - x0), abs(y1 - y0)) or 1
    for i in range(n + 1):
        x = round(x0 + (x1 - x0) * i / n)
        y = round(y0 + (y1 - y0) * i / n)
        _put(f, x, y, color)
        _put(f, x + 1, y, color)
    if node:
        mx, my = round((x0 + x1) / 2), round((y0 + y1) / 2)
        _put(f, mx - 1, my, _BAMBOO_NODE)
        _put(f, mx + 2, my, _BAMBOO_NODE)


def _m_shape(f: Face, x0: int, x1: int, ytop: int, ybot: int, ymid: int, flip: bool) -> None:
    """An M (or, flipped, a W) built from four bamboo sticks."""
    xc = (x0 + x1) // 2
    _seg(f, x0, ytop, x0, ybot, _BAMBOO)
    _seg(f, x1, ytop, x1, ybot, _BAMBOO)
    if flip:                                   # W: inner peak joins the leg bottoms
        _seg(f, x0, ybot, xc, ymid, _BAMBOO)
        _seg(f, xc, ymid, x1, ybot, _BAMBOO)
    else:                                      # M: inner valley joins the leg tops
        _seg(f, x0, ytop, xc, ymid, _BAMBOO)
        _seg(f, xc, ymid, x1, ytop, _BAMBOO)


def _bamboo(f: Face, n: int) -> None:
    """Bamboo pip layouts. 6/7/8/9 use suit-specific arrangements; 1-5 use the pip grid."""
    if n == 6:                                 # two rows of three, gap between rows
        for cy in (13, 31):
            for cx in (7, 16, 25):
                _stick(f, cx, cy, _BAMBOO)
    elif n == 7:                               # the 6 layout + one stick centred above
        _stick(f, 16, 7, _BAMBOO, half=5)
        for cy in (21, 33):
            for cx in (7, 16, 25):
                _stick(f, cx, cy, _BAMBOO, half=5)
    elif n == 8:                               # two M's, one inverted, stacked
        _m_shape(f, 6, 26, 6, 18, 10, flip=True)     # inverted M (W) on top
        _m_shape(f, 6, 26, 26, 38, 34, flip=False)   # M below
    elif n == 9:                               # three rows of three, gaps between rows
        for cy in (10, 22, 34):
            for cx in (7, 16, 25):
                _stick(f, cx, cy, _BAMBOO, half=5)
    else:
        _pips(f, n, _BAMBOO, stick=True)


def _circles(f: Face, n: int) -> None:
    """Circle pip layouts. 7/8 use suit-specific arrangements (7 keeps the
    traditional green diagonal over a red square); others use the pip grid."""
    if n == 7:                                 # green diagonal trio over a red 2x2 square
        for cx, cy in ((8, 7), (16, 14), (24, 21)):
            _ring(f, cx, cy, _CIRCLE)
        for cx, cy in ((11, 29), (21, 29), (11, 38), (21, 38)):
            _ring(f, cx, cy, _CIRCLE)
    elif n == 8:                               # two columns of four
        for cx in (12, 20):
            for cy in (8, 18, 28, 38):
                _ring(f, cx, cy, _CIRCLE)
    else:
        _pips(f, n, _CIRCLE, stick=False)


def _frame_mark(f: Face, color: str) -> None:
    """Hollow rectangle — used for the white dragon (blank/frame tile)."""
    x0, y0, x1, y1 = 6, 8, FW - 7, FH - 9
    for x in range(x0, x1 + 1):
        _put(f, x, y0, color); _put(f, x, y1, color)
    for y in range(y0, y1 + 1):
        _put(f, x0, y, color); _put(f, x1, y, color)


# ---------------------------------------------------------------------------
# Real CJK symbols, rasterised from the Mahjong Unicode glyphs
# ---------------------------------------------------------------------------

# Bonus-tile glyph codepoints (the Unicode order differs from our id order).
# Flowers 34-37 are the seasons 春夏秋冬; seasons 38-41 are the plants 梅蘭菊竹.
_SEASON_CP = {38: 0x1F022, 39: 0x1F023, 40: 0x1F025, 41: 0x1F024}   # plum/orchid/chrys/bamboo
_ANIMAL_CP = {42: 0x1F408, 43: 0x1F401, 44: 0x1F413, 45: 0x1F41B}   # cat/mouse/cockerel/worm(bug)


def _glyph(tid: int):
    """(codepoint, has_tile_frame) for a tile's glyph, or None.

    Mahjong-tile glyphs (chars/winds/dragons/flowers/seasons) carry their own tile frame,
    which we strip; animal emoji are bare shapes, so no strip.
    """
    if 9 <= tid < 18:   return (0x1F007 + (tid - 9), True)    # characters → 🀇..🀏
    if 27 <= tid < 31:  return (0x1F000 + (tid - 27), True)   # winds → 🀀..🀃
    if 31 <= tid < 34:  return (0x1F004 + (tid - 31), True)   # dragons → 🀄🀅🀆
    if 34 <= tid < 38:  return (0x1F026 + (tid - 34), True)   # flowers → 春夏秋冬
    if 38 <= tid < 42:  return (_SEASON_CP[tid], True)        # seasons → 梅蘭菊竹
    if 42 <= tid < 46:  return (_ANIMAL_CP[tid], False)       # animals → emoji
    return None


@lru_cache(maxsize=64)
def _symbol_mask(tid: int):
    """Boolean mask (_SYM_H x _SYM_W) of the tile's inner symbol, or None."""
    g = _glyph(tid)
    if g is None:
        return None
    cp, framed = g
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(_FONT_PATH, 160)
    except Exception:
        return None
    img = Image.new("L", (260, 260), 0)
    ImageDraw.Draw(img).text((40, 20), chr(cp), fill=255, font=font)
    bb = img.getbbox()
    if not bb:
        return None
    img = img.crop(bb)
    iw, ih = img.size
    if framed:
        # Strip the glyph's own tile-frame. The frame is roughly uniform in pixels, so as
        # a fraction it's smaller vertically (the glyph is taller than wide); a symmetric
        # 16% used to clip the numeral's top and 萬's feet — crop less, especially in y.
        dx, dy = int(iw * 0.12), int(ih * 0.06)
        img = img.crop((dx, dy, iw - dx, ih - dy))
    inner = img
    # Fit the symbol into the face area preserving aspect ratio, centred.
    # Character tiles keep their full numeral-over-萬 stack (legible at 32x44).
    iw2, ih2 = inner.size
    scale = min(_SYM_W / iw2, _SYM_H / ih2)
    rw, rh = max(1, int(iw2 * scale)), max(1, int(ih2 * scale))
    inner = inner.resize((rw, rh))
    px = inner.load()
    ox, oy = (_SYM_W - rw) // 2, (_SYM_H - rh) // 2
    mask = [[False] * _SYM_W for _ in range(_SYM_H)]
    for y in range(rh):
        for x in range(rw):
            if px[x, y] > 45:
                mask[oy + y][ox + x] = True
    return tuple(tuple(row) for row in mask)


def _stamp_symbol(f: Face, tid: int, color: str) -> None:
    mask = _symbol_mask(tid)
    ox, oy = (FW - _SYM_W) // 2, (FH - _SYM_H) // 2
    drew = False
    if mask is not None:
        for y in range(_SYM_H):
            for x in range(_SYM_W):
                if mask[y][x]:
                    _put(f, ox + x, oy + y, color)
                    drew = True
    if drew:
        return
    # white dragon (empty glyph) or a missing bonus glyph → hollow frame; else Latin fallback
    if tid == 33 or tid >= 34:
        _frame_mark(f, color)
    else:
        _latin(f, _fallback_char(tid), color)


def _fallback_char(tid: int) -> str:
    if 9 <= tid < 18:   return str(tid - 9 + 1)
    if 27 <= tid < 31:  return "ESWN"[tid - 27]
    return "RGW"[tid - 31]


def _latin(f: Face, ch: str, color: str) -> None:
    rows = _FALLBACK[ch]
    scale = 6
    ox = (FW - 3 * scale) // 2
    oy = (FH - 5 * scale) // 2
    for r, bits in enumerate(rows):
        for col, b in enumerate(bits):
            if b == "1":
                for yy in range(scale):
                    for xx in range(scale):
                        _put(f, ox + col * scale + xx, oy + r * scale + yy, color)


def _corner_num(f: Face, n: int, color: str, scale: int = 2) -> None:
    """A small digit in the top-right corner (flowers/seasons index)."""
    rows = _FALLBACK[str(n)]
    ox, oy = FW - 2 - 3 * scale, 3
    for r, bits in enumerate(rows):
        for col, b in enumerate(bits):
            if b == "1":
                for yy in range(scale):
                    for xx in range(scale):
                        _put(f, ox + col * scale + xx, oy + r * scale + yy, color)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_face(tid: int) -> Face:
    f = _blank_face()
    if tid < 9:
        _bamboo(f, tid + 1)
    elif tid < 18:
        _stamp_symbol(f, tid, _CHAR)
    elif tid < 27:
        _circles(f, tid - 18 + 1)
    elif tid < 31:
        _stamp_symbol(f, tid, _GOLD)
    elif tid < 34:
        _stamp_symbol(f, tid, _DRAGON[tid])
    else:
        _stamp_symbol(f, tid, _BONUS_COLOR[tid])
        if 34 <= tid < 38:
            _corner_num(f, tid - 33, _CHAR)      # flowers: red 1-4
        elif 38 <= tid < 42:
            _corner_num(f, tid - 37, _CIRCLE)    # seasons: blue 1-4
    return f


def face_to_text(face: Face, felt: str = FELT) -> list[Text]:
    """Render a face to half-block Text lines (2 vertical pixels per row)."""
    lines: list[Text] = []
    for cy in range((FH + 1) // 2):
        line = Text()
        for cx in range(FW):
            top = face[2 * cy][cx] if 2 * cy < FH else None
            bot = face[2 * cy + 1][cx] if 2 * cy + 1 < FH else None
            if top is None and bot is None:
                line.append(" ")
            else:
                line.append("▀", Style(color=top or felt, bgcolor=bot or felt))
        lines.append(line)
    return lines
