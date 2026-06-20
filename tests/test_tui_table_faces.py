"""Smoke test for the all-custom-faces table prototype (skips without Textual)."""

import pytest

pytest.importorskip("textual")

from cracked import tui_table_faces as tf  # noqa: E402


def test_zones_render_as_faces():
    # Every zone renderable should contain half-block face pixels.
    assert "▀" in tf._you_renderable().plain
    assert "▀" in tf._top_renderable().plain
    assert "▀" in tf._side_renderable(tf.LEFT).plain
    assert "▀" in tf._center_renderable().plain


def test_app_constructs():
    assert tf.TableFacesApp() is not None
