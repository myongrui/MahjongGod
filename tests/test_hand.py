import pytest
import numpy as np
from cracked.hand import HandState, Meld, MeldType
from cracked.tiles import tile_id


def test_add_remove_tile():
    h = HandState()
    h.add_tile(tile_id("b1"))
    h.add_tile(tile_id("b1"))
    assert h.concealed[tile_id("b1")] == 2
    h.remove_tile(tile_id("b1"))
    assert h.concealed[tile_id("b1")] == 1

def test_remove_tile_not_present():
    h = HandState()
    with pytest.raises(ValueError):
        h.remove_tile(tile_id("b1"))

def test_concealed_after_discard_does_not_mutate():
    h = HandState.from_tile_names(["b1", "b2", "b3"])
    original = h.concealed.copy()
    h.concealed_after_discard(tile_id("b1"))
    np.testing.assert_array_equal(h.concealed, original)

def test_total_concealed():
    h = HandState.from_tile_names(["b1", "b2", "b3", "c1", "c2"])
    assert h.total_concealed == 5

def test_meld_count():
    h = HandState()
    assert h.meld_count == 0
    h.add_meld(Meld(MeldType.PONG, (tile_id("rd"), tile_id("rd"), tile_id("rd"))))
    assert h.meld_count == 1

def test_expected_concealed():
    h = HandState()
    assert h.expected_concealed == 14
    h.add_meld(Meld(MeldType.PONG, (tile_id("rd"), tile_id("rd"), tile_id("rd"))))
    assert h.expected_concealed == 11

def test_from_tile_names():
    h = HandState.from_tile_names(["b1", "b2", "b3", "ew", "ew"])
    assert h.concealed[tile_id("b1")] == 1
    assert h.concealed[tile_id("ew")] == 2

def test_copy_is_independent():
    h = HandState.from_tile_names(["b1", "b2"])
    c = h.copy()
    c.add_tile(tile_id("b3"))
    assert h.concealed[tile_id("b3")] == 0
