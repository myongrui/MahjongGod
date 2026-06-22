"""
Smoke tests for the Ursina 3D game front-end.

Like `test_ursina_table.py`, the module is written to import with or without Ursina, so
these run everywhere and don't need a GPU/window: a structural check, plus the headless
engine loop the `GameDriver` pumps (driving a full all-AI match to completion).
"""


def test_module_imports():
    import cracked.ursina_game as g

    assert isinstance(g._HAVE_URSINA, bool)
    assert callable(g.main)
    assert g.WINDS == [27, 28, 29, 30] and g.BEAT >= 0
    if g._HAVE_URSINA:
        assert hasattr(g, "GameDriver") and hasattr(g, "Menu")


def test_spectator_match_runs_to_completion():
    # the exact step()/finish_hand loop the GameDriver drives, headless (no Ursina window)
    from cracked.match import GameMatch

    match = GameMatch(n_rounds=1, seed=7)
    hands = 0
    while not match.is_complete and hands < 60:
        match.start_hand()
        engine = match.engine
        guard = 0
        while not engine.is_finished and guard < 4000:
            engine.step()
            guard += 1
        assert engine.is_finished, "hand did not finish within the step guard"
        match.finish_hand()
        hands += 1

    assert match.is_complete
    assert match.history
