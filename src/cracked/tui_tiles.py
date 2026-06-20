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


def _stick(f: Face, cx: int, cy: int, color: str) -> None:
    for dy in range(-6, 6):
        _put(f, cx, cy + dy, color)
        _put(f, cx + 1, cy + dy, color)
    for dx in (-1, 2):
        _put(f, cx + dx, cy, _BAMBOO_NODE)
        _put(f, cx + dx, cy - 1, _BAMBOO_NODE)


def _pips(f: Face, n: int, color: str, stick: bool) -> None:
    for col, row in _PIPS[n]:
        cx, cy = _PX[col], _PY[row]
        (_stick if stick else _ring)(f, cx, cy, color)


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

def _mahjong_cp(tid: int) -> Optional[int]:
    if 9 <= tid < 18:   return 0x1F007 + (tid - 9)    # characters → 🀇..🀏
    if 27 <= tid < 31:  return 0x1F000 + (tid - 27)   # winds → 🀀..🀃
    if 31 <= tid < 34:  return 0x1F004 + (tid - 31)   # dragons → 🀄🀅🀆
    return None


@lru_cache(maxsize=64)
def _symbol_mask(tid: int):
    """Boolean mask (_SYM_H x _SYM_W) of the tile's inner symbol, or None."""
    cp = _mahjong_cp(tid)
    if cp is None:
        return None
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
    dx, dy = int(iw * 0.16), int(ih * 0.16)     # strip the glyph's own frame
    inner = img.crop((dx, dy, iw - dx, ih - dy))
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
    # white dragon (empty glyph) → hollow frame; otherwise Latin fallback
    if tid == 33:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_face(tid: int) -> Face:
    f = _blank_face()
    if tid < 9:
        _pips(f, tid + 1, _BAMBOO, stick=True)
    elif tid < 18:
        _stamp_symbol(f, tid, _CHAR)
    elif tid < 27:
        _pips(f, tid - 18 + 1, _CIRCLE, stick=False)
    elif tid < 31:
        _stamp_symbol(f, tid, _GOLD)
    else:
        _stamp_symbol(f, tid, _DRAGON[tid])
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
