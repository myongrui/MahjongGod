"""CLI integration tests using Click's CliRunner."""

import pytest
from click.testing import CliRunner

from cracked.cli import cli
from cracked.tiles import tile_id, Wind, FLOWER_SPRING, ANIMAL_CAT


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("CRACKED_STATE_FILE", str(tmp_path / "game.json"))
    return CliRunner()


def _new_game(runner, seat="east", prevailing="east"):
    return runner.invoke(cli, ["new-game", "--seat", seat, "--prevailing", prevailing])


# ---------------------------------------------------------------------------
# new-game
# ---------------------------------------------------------------------------

def test_new_game_creates_state(runner):
    r = _new_game(runner)
    assert r.exit_code == 0
    assert "New game started" in r.output


def test_new_game_seat_appears_in_output(runner):
    r = _new_game(runner, seat="south")
    assert r.exit_code == 0
    assert "South" in r.output


def test_new_game_invalid_seat(runner):
    r = runner.invoke(cli, ["new-game", "--seat", "purple"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# hand
# ---------------------------------------------------------------------------

def test_hand_set_13_tiles(runner):
    _new_game(runner)
    r = runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    assert r.exit_code == 0
    assert "Hand set" in r.output


def test_hand_wrong_tile_count_rejected(runner):
    _new_game(runner)
    r = runner.invoke(cli, ["hand", "b1", "b2"])
    assert r.exit_code != 0


def test_hand_invalid_tile_name_rejected(runner):
    _new_game(runner)
    r = runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","xx"])
    assert r.exit_code != 0


def test_hand_shows_shanten(runner):
    _new_game(runner)
    r = runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    assert "Shanten" in r.output


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

def test_draw_adds_tile(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["draw", "rd"])
    assert r.exit_code == 0
    assert "Drew" in r.output
    assert "14 tiles" in r.output


def test_draw_invalid_tile(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["draw", "xx"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# discard
# ---------------------------------------------------------------------------

def test_discard_own_tile(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    runner.invoke(cli, ["draw", "rd"])
    r = runner.invoke(cli, ["discard", "rd"])
    assert r.exit_code == 0
    assert "discarded" in r.output


def test_discard_by_opponent(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["discard", "c5", "--by", "south"])
    assert r.exit_code == 0
    assert "South" in r.output


def test_discard_invalid_tile(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["discard", "zz"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# meld
# ---------------------------------------------------------------------------

def test_meld_pong_by_opponent(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["meld", "pong", "rd", "rd", "rd", "--by", "south"])
    assert r.exit_code == 0
    assert "PONG" in r.output


def test_meld_chow_by_me(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["meld", "chow", "b1", "b2", "b3", "--by", "me"])
    assert r.exit_code == 0
    assert "CHOW" in r.output


# ---------------------------------------------------------------------------
# flower
# ---------------------------------------------------------------------------

def test_flower_own_seat_flower(runner):
    _new_game(runner, seat="east")
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["flower", "f1"])
    assert r.exit_code == 0
    assert "Spring" in r.output


def test_flower_by_opponent(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["flower", "cat", "--by", "south"])
    assert r.exit_code == 0
    assert "South" in r.output


def test_flower_invalid_name(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["flower", "tulip"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

def test_recommend_requires_14_tiles(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    # 13 tiles — no draw yet
    r = runner.invoke(cli, ["recommend"])
    assert r.exit_code == 0
    assert "Need" in r.output  # prompts user to draw first


def test_recommend_after_draw(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    runner.invoke(cli, ["draw", "rd"])
    r = runner.invoke(cli, ["recommend"])
    assert r.exit_code == 0
    assert "Discard" in r.output or "Tenpai" in r.output or "Complete" in r.output


def test_recommend_tenpai_hand(runner):
    # Hand that is tenpai after draw: pure sequences waiting on pair
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    runner.invoke(cli, ["draw", "rd"])
    r = runner.invoke(cli, ["recommend"])
    assert r.exit_code == 0
    assert "Tenpai" in r.output or "shanten: -1" in r.output or "Discard" in r.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_shows_hand(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["status"])
    assert r.exit_code == 0
    assert "Your hand" in r.output


def test_status_shows_opponents(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    r = runner.invoke(cli, ["status"])
    assert "South" in r.output
    assert "West" in r.output
    assert "North" in r.output


def test_status_shows_opponent_discards(runner):
    _new_game(runner)
    runner.invoke(cli, ["hand",
        "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd"])
    runner.invoke(cli, ["discard", "c5", "--by", "south"])
    r = runner.invoke(cli, ["status"])
    assert r.exit_code == 0
    assert "Discards" in r.output


# ---------------------------------------------------------------------------
# No active game
# ---------------------------------------------------------------------------

def test_commands_fail_without_new_game(runner):
    r = runner.invoke(cli, ["hand", "b1"])
    assert r.exit_code != 0
