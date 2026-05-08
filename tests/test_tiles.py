import numpy as np
import pytest
from cracked.tiles import (
    tile_id, tile_name, is_suited, is_honor, is_terminal,
    suit_of, rank_of, tiles_from_names, names_from_array,
    BAMBOO_START, CHAR_START, CIRCLE_START, WIND_START, DRAGON_START,
)


def test_tile_id_bamboo():
    assert tile_id("b1") == 0
    assert tile_id("b9") == 8

def test_tile_id_characters():
    assert tile_id("c1") == 9
    assert tile_id("c9") == 17

def test_tile_id_circles():
    assert tile_id("d1") == 18
    assert tile_id("d9") == 26

def test_tile_id_winds():
    assert tile_id("ew") == 27
    assert tile_id("sw") == 28
    assert tile_id("ww") == 29
    assert tile_id("nw") == 30

def test_tile_id_dragons():
    assert tile_id("rd") == 31
    assert tile_id("gd") == 32
    assert tile_id("wd") == 33

def test_tile_name_roundtrip():
    for tid in range(34):
        assert tile_id(tile_name(tid)) == tid

def test_tile_id_invalid():
    with pytest.raises(ValueError):
        tile_id("x5")
    with pytest.raises(ValueError):
        tile_id("b0")

def test_is_suited():
    assert is_suited(0)
    assert is_suited(17)
    assert is_suited(26)
    assert not is_suited(27)
    assert not is_suited(33)

def test_is_honor():
    assert not is_honor(26)
    assert is_honor(27)
    assert is_honor(33)

def test_is_terminal():
    assert is_terminal(tile_id("b1"))
    assert is_terminal(tile_id("b9"))
    assert not is_terminal(tile_id("b5"))
    assert not is_terminal(tile_id("ew"))

def test_suit_of():
    assert suit_of(tile_id("b5")) == 0
    assert suit_of(tile_id("c5")) == 1
    assert suit_of(tile_id("d5")) == 2
    with pytest.raises(ValueError):
        suit_of(27)

def test_rank_of():
    assert rank_of(tile_id("b3")) == 3
    assert rank_of(tile_id("c9")) == 9

def test_tiles_from_names():
    arr = tiles_from_names(["b1", "b1", "b2"])
    assert arr[tile_id("b1")] == 2
    assert arr[tile_id("b2")] == 1
    assert arr.sum() == 3

def test_names_from_array():
    arr = tiles_from_names(["b1", "b1", "c5"])
    names = names_from_array(arr)
    assert names == ["b1", "b1", "c5"]
