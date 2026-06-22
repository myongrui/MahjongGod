# CLAUDE.md
1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

    State your assumptions explicitly. If uncertain, ask.
    If multiple interpretations exist, present them - don't pick silently.
    If a simpler approach exists, say so. Push back when warranted.
    If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

    No features beyond what was asked.
    No abstractions for single-use code.
    No "flexibility" or "configurability" that wasn't requested.
    No error handling for impossible scenarios.
    If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

    Don't "improve" adjacent code, comments, or formatting.
    Don't refactor things that aren't broken.
    Match existing style, even if you'd do it differently.
    If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

    Remove imports/variables/functions that YOUR changes made unused.
    Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.
4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

    "Add validation" → "Write tests for invalid inputs, then make them pass"
    "Fix the bug" → "Write a test that reproduces it, then make it pass"
    "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

The following provides guidance to Claude Code when working with code in this repository.

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
python -m pytest tests/test_tiles_away.py -v

# Run a specific test
python -m pytest tests/test_tiles_away.py::test_waiting_single_wait -v

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

All 8 phases are complete. The stack is: heuristic engine (phases 1–5) → Monte Carlo simulator (phase 6) → supervised DangerNet (phase 7) → self-play RL ActorCritic (phase 8). The ActorCritic now trains **inside the real engine** over full 16-hand matches (`GameMatch`): the agent occupies one rotating seat against three fixed-weight `HeuristicPolicy` opponents, real `calculate_tai` scoring drives chip payments, and the episode reward is the agent's net chip change over the match.

### Tile Encoding (`src/cracked/tiles.py`)

Tiles are encoded as integers 0–33 in a **34-element numpy int8 array** where the value is the count of that tile type. This array representation is used throughout — not OOP tile objects — because tiles-away calculation runs thousands of times per recommendation.

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

### Tiles-away Calculator (`src/cracked/tiles_away.py`)

Three winning forms are evaluated and the minimum tiles-away count taken:
1. **Standard form**: recursive backtracking over suits + honor handling
2. **Seven pairs**: `6 - count_of_pairs`
3. **Thirteen orphans**: `13 - unique_orphans - has_pair`

Key correctness invariant: **isolated honor tiles (count=1) contribute 0 to the partial-block count** in standard form. Honor tiles can only form triplets (no sequences), so a lone honor needs 2 more draws to be useful — it is not a 1-draw partial. Only honor pairs (count≥2) count as partials.

`acceptance_count(hand13, unknown_tiles)` returns a dict of `{tile_id: count_remaining}` for tiles that reduce tiles away if drawn. Returns `{}` immediately for complete hands (tiles_away=-1).

`best_discards(hand14, unknown_tiles)` evaluates every candidate discard from a 14-tile hand, returning results sorted by `(tiles_away_after ASC, weighted_acceptance DESC)`.

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
| `python -m cracked.ursina_game` | 3D game — spectator mode (requires `.[threed]`) |

`recommend` flags: `--deep` runs Monte Carlo simulation and logs results; `--model PATH` shows DangerNet predictions (requires torch).

Tile colors in terminal output: bamboo=green, characters=red, circles=cyan, winds=yellow, dragons=magenta.

CLI tests use Click's `CliRunner` with `monkeypatch` to isolate state files — never use ad-hoc shell commands to test CLI behavior.

### Game Engine (`src/cracked/engine.py`)

`GameEngine` is a synchronous turn-by-turn state machine that owns all 4 hands, the wall, and discard piles. It drives the 3D game (`ursina_game`) and, via `GameMatch`, the self-play training loop.

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

**Claim priority**: Ron (any discard completes hand) > Pong/Kong (clockwise) > Chow (left player only). The engine owns priority, option enumeration (`_find_chow_options`), and claim *execution*; the *decision* to claim belongs to the seat's policy. Ron is always taken by the engine (not a policy choice).

**Seat policies** (`src/cracked/policy.py`): each seat is driven by a `Policy` — `choose_discard(view)` (return a tile, or `None` to await external input) plus `wants_pong` / `wants_kong` / `choose_chow`. `GameEngine(policies=...)` overrides per seat; otherwise human seats get `HumanPolicy` (defers discard to `submit_discard`, never claims) and the rest get `HeuristicPolicy` — the fixed-weight risk-aware bot backed by `recommend_discard` (danger + opponent modeling), with claim rules that maintain/improve tiles away (kong allows one extra step; chow requires strict improvement). `ModelPolicy` (`training/policy_model.py`, torch-lazy) slots a trained ActorCritic into a seat; it picks discards from the policy net and currently delegates claims to a heuristic placeholder.

### Module Dependency Order

```
tiles → hand → tiles_away → scoring → game_state → danger → opponent_model → optimizer → cli
                                                                        ↘ simulator ↗
                                                          optimizer → policy → engine → match
                                                                        optimizer → tui
                              tiles → tui_tiles → ursina_table → ursina_game   (3D game; reads match/engine)
                                              training/features → training/model → training/trainer
                                  match + policy + training/features → training/self_play
                                              training/features + policy → training/policy_model
```

### Match Manager (`src/cracked/match.py`)

`GameMatch` wraps `GameEngine` to manage a full multi-hand Singapore Mahjong match:
- Chip balances (500 starting) persist across hands — injected into `GameEngine` after `deal()` to override its reset
- Seat rotation: South→East (new dealer), East→North, North→West, West→South when a non-East player wins
- No rotation on East win or wall-exhausted draw, UNLESS a kong was declared during the hand (`engine.kong_declared`)
- After 4 rotations the table wind advances E→S→W→N; match ends after `n_rounds` table-wind rounds (default 4)
- `player_at: dict[int, int]` maps current wind constant → persistent player number (1–4); rotates with winds

Chip payment scale: shooter pays 4/8/16/32/64 chips; zimo (self-draw) each opponent pays 2/4/8/16/32 chips (tai 1–5).

### Singapore Mahjong Specifics

- Minimum **1** tai to win (not 3)
- Shooter-pays-all: the player who discards the winning tile pays all losers
- Tai scoring is exponential (payment = base × 2^tai), making high-tai hands disproportionately dangerous
- Chow only from the player to your left
- Seven pairs is a valid winning form
- Wall stops at **15 tiles remaining** (dead wall) — drawing stops there, not at 0
- Kong during a hand sets `engine.kong_declared = True`; if wall then exhausts, seats rotate (unlike a normal draw)

### Training Feature Vectors (`src/cracked/training/features.py`)

Two vector layouts are derived from a `GameState` (constants are the source of truth — keep these in sync if the layout changes):

- `extract_state_features(state)` → **`N_STATE_FEATURES = 230`** (89-dim state block + 3 × 47-dim opponent blocks). Used by the ActorCritic policy/value nets in `self_play.py`, which evaluate the whole state at once.
- `extract_features(state, candidate_discard)` → **`N_FEATURES = 265`** (the 230 state features plus a 35-dim candidate-discard block: 34-dim tile one-hot + tiles-away-after). Used by the supervised `DangerNet`, which scores one candidate discard at a time.

The 89-dim state block covers concealed/unknown tile counts, meld/flower/animal counts, tiles away, seat and prevailing wind one-hots, wall/turn progress, and optimizer-derived hand-structure signals (tai potential, flush purity, pair count, all-pong/seven-pairs flags, adaptive α). Each 47-dim opponent block covers seat one-hot, meld count, discard counts, waiting probability, suit/honor bias, and dragon/wind danger counts.
