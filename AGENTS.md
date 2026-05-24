# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Install (core + dev)
pip install -e ".[dev]"

# Install with PyTorch for ML training
pip install -e ".[dev,ml]"

# Install with Textual TUI
pip install -e ".[dev,ui]"

# Run all fast tests (slow simulation tests excluded by default)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_shanten.py -v

# Run a specific test
python -m pytest tests/test_shanten.py::test_tenpai_single_wait -v

# Run slow tests (full game simulations)
python -m pytest -m slow -v

# Play a game session
cracked new-game --seat east --prevailing east
cracked hand b1 b2 b3 c1 c2 c3 d1 d2 d3 ew ew ew rd
cracked draw gd
cracked recommend              # heuristic
cracked recommend --deep --games 200  # Monte Carlo (logs to data/game_log.jsonl)
cracked recommend --model models/danger_net.pt  # trained model
cracked status

# Train DangerNet (supervised) after accumulating --deep data
python -m cracked.training.trainer --log data/game_log.jsonl --out models/danger_net.pt

# Train ActorCritic (self-play RL)
python -m cracked.training.self_play --episodes 5000 --out models/policy.pt
```

## Architecture

All 8 phases are complete. The stack is: heuristic engine (phases 1–5) → Monte Carlo simulator (phase 6) → supervised DangerNet (phase 7) → self-play RL ActorCritic (phase 8).

### Tile Encoding (`src/cracked/tiles.py`)

Tiles are encoded as integers 0–33 in a **34-element numpy int8 array** where the value is the count of that tile type. This array representation is used throughout — not OOP tile objects — because shanten calculation runs thousands of times per recommendation.

```
0–8:   Bamboo 1–9
9–17:  Characters 1–9
18–26: Circles 1–9
27–30: Winds (East, South, West, North)
31–33: Dragons (Red, Green, White)
```

Flowers, seasons, and animals have IDs 34–45 and are **tracked separately** in `HandState` — they are never part of the 34-array hand composition, only affect scoring.

Bonus tile helpers: `bonus_tile_id(name)` parses names like `"f1"`, `"spring"`, `"cat"`; `bonus_tile_name(bid)` returns display labels; `is_animal(bid)` distinguishes animals from flowers/seasons.

### Hand State (`src/cracked/hand.py`)

`HandState` holds:
- `concealed`: 34-element array of concealed tile counts
- `melds`: list of exposed `Meld` objects (chow/pong/kong)
- `flowers`, `animals`: bonus tile IDs set aside on draw
- `seat_wind`: tile ID of player's seat wind

`Meld` stores `MeldType` (CHOW/PONG/KONG), the tile IDs, whether it's concealed, and the source player seat.

### Shanten Calculator (`src/cracked/shanten.py`)

Three winning forms are evaluated and the minimum shanten taken:
1. **Standard form**: recursive backtracking over suits + honor handling
2. **Seven pairs**: `6 - count_of_pairs`
3. **Thirteen orphans**: `13 - unique_orphans - has_pair`

Key correctness invariant: **isolated honor tiles (count=1) contribute 0 to the partial-block count** in standard form. Honor tiles can only form triplets (no sequences), so a lone honor needs 2 more draws to be useful — it is not a 1-draw partial. Only honor pairs (count≥2) count as partials.

`acceptance_count(hand13, unknown_tiles)` returns a dict of `{tile_id: count_remaining}` for tiles that reduce shanten if drawn. Returns `{}` immediately for complete hands (shanten=-1).

`best_discards(hand14, unknown_tiles)` evaluates every candidate discard from a 14-tile hand, returning results sorted by `(shanten_after ASC, weighted_acceptance DESC)`.

### Scoring Engine (`src/cracked/scoring.py`)

`calculate_tai(hand, ctx, rules)` is the main entry point. It:
1. Checks for limit hands first (thirteen orphans, all honors, big three dragons, four winds, all kongs)
2. Finds all valid decompositions of the concealed tiles via recursive backtracking
3. Scores each decomposition and returns the highest-scoring result (capped at `HouseRules.tai_cap`)
4. Adds flower/season tai on top of the cap (configurable)

`WinContext` carries win circumstances (self-draw, last tile, robbing kong, prevailing wind).
`HouseRules` is configurable: tai cap (default 5), minimum tai (default 1), seven-pairs base (2 or 3), flower-above-cap flag.

Key scoring elements: dragon pongs (1 each), wind pongs (1 each, +1 if double wind), all pongs (2), half flush (2), full flush (4), small three dragons (3), seven pairs (3 base), limit hands (cap).

Flower/season rules: only the tile matching your seat gives 1 tai (e.g. East seat → Spring flower or Plum season). Non-matching flowers/seasons give 0 tai. Each animal always gives 1 tai regardless of seat.

### Game State (`src/cracked/game_state.py`)

`GameState` is the top-level observable state for one session:
- `my_hand`: `HandState` for the player
- `opponents`: list of three `PlayerView` objects (one per opponent seat)
- `wall_tiles_remaining`, `turn_number`: game progress counters

`PlayerView` tracks an opponent's `discards`, `melds`, and `flowers` (bonus tiles) by seat wind.

Key methods:
- `visible_tiles()` — union of our hand + all exposed melds + all discards
- `unknown_tiles()` — `clip(full_wall - visible_tiles, 0, 4)`; used as input to `best_discards()`
- `opponent_by_seat(seat)` — look up opponent by wind constant

State is persisted to `.cracked_game.json` (or `$CRACKED_STATE_FILE` if set) via `save_state()`/`load_state()`. The env var is used in tests to isolate state files via `tmp_path`.

### CLI (`src/cracked/cli.py`)

Built with Click + Rich. All commands load/save state from the JSON file.

| Command | Description |
|---|---|
| `cracked new-game [--seat] [--prevailing]` | Start a new session |
| `cracked hand TILES...` | Set your 13-tile starting hand |
| `cracked draw TILE` | Record a tile drawn from the wall |
| `cracked discard TILE [--by SEAT]` | Record a discard (yours or opponent's) |
| `cracked meld TYPE TILES... --by SEAT [--concealed]` | Record a pong/kong/chow |
| `cracked flower TILE [--by SEAT]` | Record a bonus tile reveal |
| `cracked recommend [--deep] [--games N] [--log FILE] [--model PATH]` | Discard recommendations |
| `cracked status` | Display the full game state |
| `cracked-ui` | Textual TUI advisor mode (requires `.[ui]`) |
| `cracked-play` | Textual TUI game viewer — spectator or interactive (requires `.[ui]`) |

`recommend` flags: `--deep` runs Monte Carlo simulation and logs results; `--model PATH` shows DangerNet predictions (requires torch).

Tile colors in terminal output: bamboo=green, characters=red, circles=cyan, winds=yellow, dragons=magenta.

CLI tests use Click's `CliRunner` with `monkeypatch` to isolate state files — never use ad-hoc shell commands to test CLI behavior.

### Game Engine (`src/cracked/engine.py`)

`GameEngine` is a synchronous turn-by-turn state machine that owns all 4 hands, the wall, and discard piles. It drives both TUI modes.

**Wall**: 148 tiles — 136 standard (4 copies each of 34 types) + 12 bonus tiles (flowers 34-37, seasons 38-41, animals 42-45). Bonus tiles drawn during play are set aside with a replacement drawn automatically.

**Event types** emitted by `deal()`, `step()`, and `submit_discard()`:

| EventType | When |
|---|---|
| `DEAL` | Game starts — wall shuffled, 13 tiles dealt |
| `DRAW` | A standard tile drawn from the wall |
| `BONUS` | A bonus tile set aside; replacement follows |
| `DISCARD` | A tile discarded |
| `MELD` | A pong/kong/chow claimed from a discard |
| `WIN_SELF_DRAW` | Player completes hand on their own draw |
| `WIN_DISCARD` | Player completes hand on an opponent's discard |
| `WALL_EXHAUSTED` | Wall empty — draw game |
| `AWAIT_DISCARD` | Human player's turn — call `submit_discard()` |

**Claim priority**: Ron (any discard completes hand) > Pong/Kong (clockwise) > Chow (left player only). Human players skip claim opportunities (no claim UI yet).

**AI claim heuristics**: `_ai_wants_pong()` accepts if best post-pong discard maintains or improves shanten; `_ai_wants_kong()` allows up to shanten+1 (replacement compensates); `_pick_best_chow()` requires strict shanten improvement.

### Module Dependency Order

```
tiles → hand → shanten → scoring → game_state → danger → opponent_model → optimizer → cli
                                                                        ↘ simulator ↗
                                                          optimizer + simulator → engine → match → tui_game
                                                                        optimizer → tui
                                              training/features → training/model → training/trainer
                                              training/features → training/self_play
```

### Singapore Mahjong Specifics

- Minimum **1** tai to win (not 3)
- Shooter-pays-all: the player who discards the winning tile pays all losers
- Tai scoring is exponential (payment = base × 2^tai), making high-tai hands disproportionately dangerous
- Chow only from the player to your left
- Seven pairs is a valid winning form
- Wall stops at **15 tiles remaining** (dead wall) — drawing stops there, not at 0
- Kong during a hand sets `engine.kong_declared = True`; if wall then exhausts, seats rotate (unlike a normal draw)
