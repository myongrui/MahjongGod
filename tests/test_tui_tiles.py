"""
Tests for the custom tile-face artwork.

These need only `rich` (a core dependency), not Textual, so they always run.
They exercise the structure of every face; the actual look is verified visually
via the gallery render / the demos. Symbol tiles use rasterised CJK when Pillow
is available and fall back to a Latin pixel font otherwise — both produce art,
so the tests stay agnostic to which path ran.
"""

from cracked.tui_tiles import FW, FH, make_face, face_to_text, _IVORY, _BORDER


def _art_pixels(tid: int) -> int:
    f = make_face(tid)
    return sum(1 for row in f for c in row if c not in (_IVORY, _BORDER, None))


def test_every_face_has_correct_dimensions():
    for tid in range(34):
        f = make_face(tid)
        assert len(f) == FH
        assert all(len(row) == FW for row in f)


def test_every_face_has_artwork():
    assert all(_art_pixels(tid) > 0 for tid in range(34))


def test_face_to_text_renders_lines():
    lines = face_to_text(make_face(0))
    assert len(lines) == (FH + 1) // 2
    assert any("▀" in line.plain for line in lines)


def test_suit_groups_distinct_within_group():
    # Within each number suit, different ranks place different pip counts.
    bamboo = {_art_pixels(t) for t in range(0, 9)}
    circles = {_art_pixels(t) for t in range(18, 27)}
    assert len(bamboo) > 3
    assert len(circles) > 3
