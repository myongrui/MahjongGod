# crackedMahjong — Singaporean Mahjong Discard Optimizer

## Context

You want a tool that recommends the best tile to discard from your hand, balancing winning probability against the risk of "shooting" (dealing into an opponent's win). Singapore Mahjong's **shooter-pays-all** rule combined with exponential tai scoring makes careless discards catastrophic — this tool aims to quantify that risk and make informed recommendations. The "slowly train" goal means we build a heuristic engine first, then layer ML on top.

## Tech Stack

- **Python 3.11+** — best ML ecosystem for the training pipeline
- **numpy** — tile representation as 34-element arrays, vectorized shanten calculation
- **PyTorch** — neural network for danger estimation (later phases)
- **click** — CLI framework
- **pydantic** — data validation for game state
- **rich** — colored tile display in terminal
- **pytest** — testing

No existing Mahjong library reuse — the `mahjong` PyPI package is riichi-specific (furiten, yaku, fu). Singapore rules differ fundamentally, so we build our own.

## Project Structure

```
crackedMahjong/
├── pyproject.toml
├── CLAUDE.md
├── src/cracked/
│   ├── tiles.py           # Tile encoding (34-type int scheme) and constants
│   ├── hand.py            # Hand state: concealed tiles, melds, flowers
│   ├── shanten.py         # Shanten calculator (tiles-to-win)
│   ├── scoring.py         # Singapore tai scoring engine
│   ├── game_state.py      # Full observable game state
│   ├── wall.py            # Remaining/unknown tile inference
│   ├── danger.py          # Danger tile assessment
│   ├── opponent_model.py  # Opponent hand/tai estimation
│   ├── optimizer.py       # Discard recommendation engine
│   ├── simulator.py       # Monte Carlo game simulation
│   ├── cli.py             # CLI interface
│   └── training/
│       ├── features.py    # Feature extraction from game state
│       ├── model.py       # Neural network architecture
│       ├── self_play.py   # Self-play game generation
│       ├── trainer.py     # Training loop
│       └── data.py        # Game log parsing/dataset management
├── models/                # Saved model weights
├── data/                  # Training data, game logs
└── tests/
    ├── test_tiles.py
    ├── test_shanten.py
    ├── test_scoring.py
    ├── test_danger.py
    ├── test_optimizer.py
    └── fixtures/          # Known hand configurations
```

## Core Design Decisions

### Tile Representation
Tiles encoded as integers 0–33 in a 34-element numpy array (index = tile type, value = count):
- 0–8: Bamboo 1–9
- 9–17: Characters 1–9
- 18–26: Circles 1–9
- 27–30: Winds (East, South, West, North)
- 31–33: Dragons (Red, Green, White)

Flowers, seasons, and animals are tracked separately (they're set aside on draw, never part of hand composition, only affect scoring).

**Why arrays over OOP?** Shanten calculation runs thousands of times per recommendation. Numpy arrays are orders of magnitude faster than tile object lists.

### The Optimizer Formula
For each candidate discard:
```
utility(tile) = α × offensive_value(tile) − (1 − α) × expected_shooting_cost(tile)
```

- **Offensive value**: shanten reduction + acceptance count (how many draws improve the hand) + potential tai
- **Expected shooting cost**: P(opponent tenpai) × P(tile completes their hand) × E[payment given tai]
- **α** adjusts dynamically: higher when our hand is strong/close, lower when opponents look dangerous or few tiles remain

### Danger Assessment (The Killer Feature)
For each opponent, estimate:
1. **Tenpai probability** — from meld count, discard patterns, turn number
2. **Faan range** — from exposed melds and discard patterns (flush signals, honor collecting, etc.)
3. **Dangerous tiles** — tiles NOT in their discards, adjacent to their melds, matching their apparent suit bias

Safe tiles: tiles the opponent already discarded (100% safe against them), tiles where all 4 copies are visible.

Since tai scoring is exponential (payment = base × 2^tai), even a small chance of dealing into a 5-tai hand dominates expected cost. The tool quantifies this.

## Phased Implementation

### Phase 1: Scaffolding + Tile Model + Shanten (Start Here)
- `pyproject.toml`, package structure, CLAUDE.md
- `tiles.py`: tile encoding, constants, display helpers
- `hand.py`: HandState dataclass with concealed array, melds, flowers
- `shanten.py`: recursive decomposition algorithm for standard form, seven pairs, thirteen orphans
- Acceptance count calculator (uke-ire)
- **Tests**: validate against known shanten values — this is the foundation everything else depends on

### Phase 2: Scoring Engine
- `scoring.py`: complete Singapore tai table (seat/prevailing wind, dragons, flush, all pongs, limit hands, flowers/animals)
- Configurable house rules (tai cap, animal variants)
- Hand decomposition to find highest-scoring interpretation
- **Tests**: every scoring element needs test cases

### Phase 3: Game State + CLI
- `game_state.py`: GameState, PlayerView dataclasses
- `wall.py`: visible/unknown tile tracking
- `cli.py`: commands for `new-game`, `hand`, `draw`, `discard`, `meld`, `flower`, basic stats display
- Stateful CLI session (tracks game state across commands)

### Phase 4: Danger Assessment
- `danger.py`: safe tile identification, suji-equivalent analysis, flush detection, danger scoring
- `opponent_model.py`: tenpai probability, tai estimation, waiting tile inference
- Expected shooting cost calculation
- **Tests**: constructed scenarios where danger should clearly flag certain tiles

### Phase 5: Heuristic Optimizer (First Usable Version)
- `optimizer.py`: combine offensive + defensive scores, adaptive α
- CLI `recommend` command with detailed output showing shanten, acceptance, danger per tile
- **This is where the tool becomes practically useful**

### Phase 6: Monte Carlo Simulation
- `simulator.py`: random deal generation, heuristic AI opponents, parallel simulation
- `recommend --deep` CLI option for simulation-backed recommendations
- Use simulation results to calibrate/validate heuristic optimizer

### Phase 7: ML Training Pipeline
- `training/features.py`: GameState → fixed-size tensor (~300–500 dims)
- `training/model.py`: residual MLP (3–4 hidden layers, 256–512 units)
- `training/trainer.py`: supervised learning on Monte Carlo-generated labels
- `training/data.py`: game log recording + dataset management
- `recommend --model` CLI option
- "Slowly train": tool logs every game state during play, periodically retrain on accumulated data

### Phase 8: Self-Play RL (Long-Term)
- `training/self_play.py`: 4-agent self-play, policy gradient updates (PPO)
- Tournament evaluation against heuristic baseline

## Verification

- **Shanten**: cross-validate against known hand-shanten pairs; test edge cases (seven pairs, thirteen orphans, hands with kongs)
- **Scoring**: test every tai element individually and in combination; test highest-scoring decomposition selection
- **Danger**: construct scenarios where specific tiles are clearly safe/dangerous and verify the engine agrees
- **Optimizer**: run against known game positions where the "correct" discard is clear to an experienced player
- **End-to-end**: play through sample games via CLI, verify state tracking and recommendations match manual analysis
- Run full test suite: `pytest tests/`

## Key References
- [Calculating Shanten — ezyang's blog](https://blog.ezyang.com/2014/04/calculating-shanten-in-mahjong/)
- [Fast Deficiency Number Algorithm (arXiv)](https://arxiv.org/abs/2108.06832)
- [Singapore Mahjong Scoring Rules — Tabletopia](https://c.tabletopia.com/games/singapore-mahjong/rules/singaporean-mahjong-scoring-rules/en)
- [Building a 3-Player Mahjong AI using Deep RL (arXiv)](https://arxiv.org/abs/2202.12847)
