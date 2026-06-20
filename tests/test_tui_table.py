"""
Smoke tests for the static table-view prototype.

The module imports Textual (via cracked.tui_game), so every test skips when the
optional `ui` extra is not installed — matching the rest of the TUI's
manual-verification posture. These assert only that the sample-data markup
builders produce non-empty, colored output; the visual layout is verified by
launching `python -m cracked.tui_table`.
"""

import pytest

pytest.importorskip("textual")

from cracked import tui_table as t  # noqa: E402


def test_center_shows_round_and_wall():
    s = t._center_text()
    assert t.ROUND_WIND in s and str(t.WALL_REMAINING) in s


def test_scatter_renders_every_discard():
    markup = t._scatter_pool(t.DISCARDS)
    # Each discard contributes one colored glyph segment (one closing tag).
    assert markup.count("[/") == len(t.DISCARDS)


def test_you_renderable_has_status_and_meld_label():
    plain = t._you_renderable().plain
    assert "WAITING" in plain
    assert "Pong" in plain  # the exposed meld label renders


def test_card_title_marks_dealer_and_wind():
    title = t._card_title(t.YOU)
    assert "DEALER" in title and t._WIND_CJK["East"] in title
    assert "DEALER" not in t._card_title(t.TOP)


def test_side_and_top_have_backs():
    # Opponent backs stay compact glyph blocks in the renderables.
    assert "█" in t._side_renderable(t.LEFT).plain
    assert "█" in t._top_renderable().plain


def test_faces_row_has_expected_line_count():
    rows = t._faces_row([0, 1, 2], cw=6, ch=4)
    assert len(rows) == 4
    assert all("▀" in line.plain for line in rows)


def test_tiles_render_as_bare_colored_glyphs():
    # Dense tiles stay suit-colored glyphs with no background fill.
    chip = t._chip(0)
    assert t.tile_glyph(0) in chip
    assert " on " not in chip


def test_app_constructs():
    app = t.TablePrototypeApp()
    assert app is not None
