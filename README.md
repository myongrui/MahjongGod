# crackedMahjong

A Singaporean Mahjong discard optimizer. Recommends which tile to discard by balancing winning probability against the risk of dealing into an opponent's winning hand (shooting). Includes a Monte Carlo simulator and a self-play RL training pipeline.

## Installation

```bash
# Core + CLI + dev tools
pip install -e ".[dev]"

# With PyTorch (for ML training and self-play)
pip install -e ".[dev,ml]"

# With Textual TUI advisor
pip install -e ".[dev,ui]"

# With the 3D game (Ursina)
pip install -e ".[dev,threed]"
```

## Tile notation

| Notation | Tile |
|---|---|
| `b1`–`b9` | Bamboo 1–9 |
| `c1`–`c9` | Characters 1–9 |
| `d1`–`d9` | Circles 1–9 |
| `ew` `sw` `ww` `nw` | East / South / West / North wind |
| `rd` `gd` `wd` | Red / Green / White dragon |
| `f1`–`f4` | Flowers (Spring / Summer / Autumn / Winter) |
| `s1`–`s4` | Seasons (Plum / Orchid / Chrysanthemum / Bamboo) |
| `a1`–`a4` | Animals (Cat / Mouse / Cockerel / Worm) |

---

## Playing a game

### 1. Start a new game

```bash
cracked new-game --seat east --prevailing east
```

### 2. Enter your starting hand (13 tiles)

```bash
cracked hand b1 b2 b3 c1 c2 c3 d1 d2 d3 ew ew ew rd
```

### 3. Draw a tile

```bash
cracked draw gd
```

### 4. Get a discard recommendation

```bash
cracked recommend
```

The heuristic table shows tiles away, acceptance tile count, danger score, and shooting cost for each candidate discard.

### 5. Record your discard

```bash
cracked discard gd
```

### 6. Record opponent actions

```bash
# Opponent discards
cracked discard b5 --by south

# Opponent makes a meld
cracked meld pong rd --by south

# Opponent reveals a bonus tile
cracked flower f2 --by west
```

### 7. Check game state at any time

```bash
cracked status
```

---

## Recommendation modes

### Heuristic (fast, default)

```bash
cracked recommend
```

Ranks discards by tiles-away reduction, acceptance tile count, and danger score. Runs instantly.

### Monte Carlo simulation (`--deep`)

```bash
cracked recommend --deep --games 200
```

Simulates 200 complete games per candidate discard. Shows Win%, Shoot%, and expected gain (E[Gain]). Also logs results to `data/game_log.jsonl` for ML training.

```bash
# Custom log location
cracked recommend --deep --games 500 --log data/my_session.jsonl
```

### Trained model (`--model`)

```bash
cracked recommend --model models/danger_net.pt
```

Shows predictions from a trained DangerNet alongside the heuristic table. Requires PyTorch.

---

## TUI advisor (`cracked-ui`)

Requires `pip install -e ".[ui]"` (adds the `textual` package).

```bash
cracked-ui
```

Enter any 13-tile hand and a draw tile. Press **R** to get ranked discard recommendations (tiles away, acceptance tiles, danger score, utility). Press **S** to run Monte Carlo simulation. Use the seat and prevailing wind dropdowns.

---

## 3D game (Ursina)

A real-time 3D mahjong table driven by the live engine. Requires `pip install -e ".[threed]"` (adds `ursina` + `pillow`).

```bash
python -m cracked.ursina_game        # Menu → Spectate (watch the AI play)
```

**Spectator** — watch four AI bots play a full match. Each seat is driven by a `Policy` (`src/cracked/policy.py`); the default `HeuristicPolicy` plays the full risk-aware heuristic (danger scoring + opponent modeling via `recommend_discard`), not just tiles-away minimization. Claim mechanics (pong, kong, chow) are live, and bonus tiles (flowers, seasons, animals) are drawn and replaced automatically.

The table renders the custom `tui_tiles` pixel-art faces as textures, with seats arranged **anti-clockwise** (E→S→W→N) as on a real table. Discards are thrown into a central pool with slide-and-collide physics; exposed melds and bonus tiles lie face-up in front of each player, facing the owner. A left panel shows the table wind, current seat, chips, and tiles left in the wall; a right panel shows a rolling action log. Drag to orbit, scroll to zoom.

> Interactive play (you at East, click-to-discard, pong/kong/chow/hu prompts) is the next pass; the menu's **Play** button is a placeholder until the engine gains a claim-await hook.
>
> `python -m cracked.ursina_table` runs the original static 3D prototype (one dealt scene, no game loop) that `ursina_game` is built on.

---

## ML training

### Step 1 — Collect training data

Play several sessions using `--deep`. Each run appends examples to `data/game_log.jsonl`.

```bash
cracked recommend --deep --games 200
```

### Step 2 — Train the supervised model (DangerNet)

```bash
python -m cracked.training.trainer \
    --log data/game_log.jsonl \
    --out models/danger_net.pt \
    --epochs 50
```

### Step 3 — Train the RL policy (ActorCritic via self-play)

```bash
python -m cracked.training.self_play \
    --episodes 20000 \
    --out models/policy.pt \
    --n-rounds 4 \
    --eval-every 500 \
    --eval-games 20 \
    --resume        # continue from existing checkpoint
```

The policy trains **inside the real game engine** over full multi-hand matches (`GameMatch`): the agent occupies one rotating seat against three fixed-weight `HeuristicPolicy` opponents, real `calculate_tai` scoring drives chip payments, and the episode reward is the agent's **net chip change over the whole match**. The agent auto-claims pongs, kongs, and chows using the heuristic placeholder (claim learning is a future step), then runs the RL policy to decide what to discard after each claim. Wall stops at 15 tiles remaining, matching real game rules. Checkpoints are saved whenever evaluation improves.

**Reward tuning flags:**

| Flag | Default | Description |
|---|---|---|
| `--n-rounds` | `4` | Table-wind rounds per match (`4` = East+South+West+North, a full game) |
| `--gamma` | `1.0` | Discount factor for returns (`0.99` = standard discounting) |
| `--tiles-away-reward` | `0.0` | Per-step reward for each tiles-away improvement |
| `--waiting-bonus` | `0.0` | One-time reward the first time the agent reaches waiting in a game |
| `--reward-scale` | `1.0` | Multiply all rewards by this factor (`0.0208` ≈ `1/48` normalises to `[-1, +1]`) |
| `--ent-coef` | `0.01` | Entropy coefficient in the PPO loss |
| `--shaping-scale` | `1.0` | PBRS potential-shaping weight (uses shanten + tai potential + acceptance count) |
| `--defense-weight` | `0.02` | Per-step penalty scaled by expected shooting cost |

### Step 4 — Compare reward systems in parallel

The experiment runner trains multiple reward variants simultaneously and prints a comparison table:

```bash
# List all built-in variants
python -m cracked.training.experiment --list

# Run all variants (one model file saved per variant under models/)
python -m cracked.training.experiment --episodes 10000

# Run specific variants
python -m cracked.training.experiment waiting normalized normed_waiting --episodes 10000

# Cap parallel workers (useful on machines with fewer cores)
python -m cracked.training.experiment --episodes 10000 --workers 2

# Resume all variants from their existing checkpoints
python -m cracked.training.experiment --episodes 10000 --resume
```

Built-in variants:

| Variant | What it tests |
|---|---|
| `baseline` | Original reward system |
| `discount` | `gamma=0.99` discounting only |
| `tiles_away_dense` | Per-step tiles-away-progress reward |
| `strong_shaping` | Higher `shaping_scale` + lower `ent_coef` |
| `full_fix` | Discount + tiles-away reward + strong shaping |
| `waiting` | One-time waiting bonus (no normalisation) |
| `normalized` | Reward normalisation to `[-1, +1]` only |
| `normed_waiting` | Normalised + waiting bonus + `full_fix` params |

Each variant saves its model to `models/{variant_name}.pt`. Add new variants by editing the `VARIANTS` dict at the top of `src/cracked/training/experiment.py`.

---

## Running tests

```bash
# Fast suite (default — excludes slow simulation tests)
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_tiles_away.py -v

# Run a specific test
python -m pytest tests/test_tiles_away.py::test_waiting_single_wait -v

# Include slow tests (full game simulations, ~30–60s extra)
python -m pytest -m slow -v

# All tests including slow
python -m pytest tests/ -m "" -v
```

### Test files

| File | What it covers |
|---|---|
| `test_tiles.py` | Tile encoding, name parsing, bonus tiles |
| `test_tiles_away.py` | Tiles-away calculator (standard, seven pairs, thirteen orphans) |
| `test_scoring.py` | Tai scoring engine, limit hands, flowers/animals |
| `test_danger.py` | Danger scoring, safe tile identification |
| `test_game_state.py` | GameState, PlayerView, visible/unknown tile tracking |
| `test_optimizer.py` | Discard recommendations, adaptive alpha, utility scoring |
| `test_cli.py` | All CLI commands via Click's CliRunner |
| `test_simulator.py` | Monte Carlo simulator, SimHand, game results |
| `test_features.py` | ML feature vector extraction (230-dim state and 265-dim candidate vectors) |
| `test_training_data.py` | JSONL log recording and dataset loading |
| `test_self_play.py` | ActorCritic network, episode collection, PPO update, tournament |

The `test_self_play.py` torch-dependent tests skip automatically if PyTorch is not installed. Install with `pip install -e ".[ml]"` to run them.

---

## Project structure

```
src/cracked/
├── tiles.py           # Tile encoding and constants
├── hand.py            # HandState, Meld, MeldType
├── tiles_away.py      # Tiles-away calculator + acceptance count
├── scoring.py         # Singapore tai scoring engine
├── game_state.py      # GameState, PlayerView, save/load
├── danger.py          # Tile danger scoring
├── opponent_model.py  # Opponent waiting/flush inference
├── optimizer.py       # Heuristic discard recommendations
├── simulator.py       # Monte Carlo game simulator
├── cli.py             # Click CLI (cracked)
├── engine.py          # Turn-by-turn game state machine (drives the game)
├── match.py           # Multi-hand match manager (chips, seat rotation)
├── policy.py          # Per-seat policies (HeuristicPolicy, HumanPolicy)
├── tui.py             # Textual TUI advisor mode (cracked-ui)
├── tui_tiles.py       # Custom pixel-art tile faces (shared by the 3D table)
├── ursina_table.py    # 3D table prototype + tile model, lighting, discard physics
├── ursina_game.py     # Live 3D game (python -m cracked.ursina_game)
└── training/
    ├── features.py    # GameState → numpy feature vectors
    ├── model.py       # DangerNet (supervised residual MLP)
    ├── data.py        # JSONL log recording and loading
    ├── trainer.py     # DangerNet training loop
    ├── self_play.py   # ActorCritic, PPO, self-play training
    └── experiment.py  # Parallel reward-variant experiment runner
```
