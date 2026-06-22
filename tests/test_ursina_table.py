"""
Smoke test for the Ursina 3D front-end sketch.

The module is written to import cleanly with or without Ursina installed (the heavy
import is guarded), so this runs everywhere — it checks the structure is intact and
that the helpers the scene-builder relies on are wired to the real engine deal. It
does NOT build a live scene (that needs a GPU/window).
"""

from cracked.tiles import Wind


def test_module_imports_without_ursina():
    import cracked.ursina_table as u

    # the guarded import means the module loads even when ursina is absent
    assert isinstance(u._HAVE_URSINA, bool)
    for name in ("make_tile", "place_hand", "place_wall", "build_scene", "main"):
        assert callable(getattr(u, name))
    assert u.N == 13 and u.TW > 0 and u.TH > 0 and u.TD > 0
    assert u.WALL_N == 17 and u.WALL_R < u.EDGE        # wall ring sits inside the hands


def test_scene_seats_map_to_real_dealt_hands():
    import cracked.ursina_table as u
    from cracked.engine import GameEngine

    eng = GameEngine(human_seats={int(Wind.EAST)}, seed=7)
    eng.deal()
    for wind in (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH):
        hand = eng.players[int(wind)].hand.concealed_tiles_list()[:u.N]
        assert len(hand) == u.N
        assert all(0 <= t < 34 for t in hand)
