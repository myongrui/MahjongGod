"""
Reward-system experiment runner.

Trains multiple variants in parallel and prints a comparison table.

Usage:
    # Run all variants
    python -m cracked.training.experiment

    # Run specific variants
    python -m cracked.training.experiment baseline full_fix

    # List available variants
    python -m cracked.training.experiment --list

    # Control parallelism and episode count
    python -m cracked.training.experiment --workers 2 --episodes 10000

    # Save models to a custom directory
    python -m cracked.training.experiment --out-dir models/exp1
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from cracked.training.self_play import (
    ENT_COEF,
    SHAPING_SCALE,
    DEFENSE_WEIGHT,
    train_self_play,
)

# ---------------------------------------------------------------------------
# Variant definitions — edit this table to add or tweak reward systems
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "baseline": {
        "description": "Original reward system (no changes)",
        "gamma": 1.0,
        "tiles_away_reward": 0.0,
        "waiting_bonus": 0.0,
        "reward_scale": 1.0,
        "shaping_scale": SHAPING_SCALE,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": ENT_COEF,
    },
    "discount": {
        "description": "Add gamma=0.99 discounting only",
        "gamma": 0.99,
        "tiles_away_reward": 0.0,
        "waiting_bonus": 0.0,
        "reward_scale": 1.0,
        "shaping_scale": SHAPING_SCALE,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": ENT_COEF,
    },
    "tiles_away_dense": {
        "description": "Explicit per-step tiles_away-progress reward",
        "gamma": 1.0,
        "tiles_away_reward": 0.3,
        "waiting_bonus": 0.0,
        "reward_scale": 1.0,
        "shaping_scale": SHAPING_SCALE,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": ENT_COEF,
    },
    "strong_shaping": {
        "description": "Higher shaping_scale + lower ent_coef",
        "gamma": 1.0,
        "tiles_away_reward": 0.0,
        "waiting_bonus": 0.0,
        "reward_scale": 1.0,
        "shaping_scale": 3.0,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": 0.001,
    },
    "full_fix": {
        "description": "All improvements: discount + tiles_away + strong shaping",
        "gamma": 0.99,
        "tiles_away_reward": 0.3,
        "waiting_bonus": 0.0,
        "reward_scale": 1.0,
        "shaping_scale": 3.0,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": 0.001,
    },
    # --- new variants testing waiting bonus and reward normalisation ---
    "waiting": {
        "description": "One-time waiting bonus (no normalization)",
        "gamma": 1.0,
        "tiles_away_reward": 0.0,
        "waiting_bonus": 4.0,
        "reward_scale": 1.0,
        "shaping_scale": SHAPING_SCALE,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": ENT_COEF,
    },
    "normalized": {
        "description": "Reward normalisation to [-1,+1] only",
        "gamma": 1.0,
        "tiles_away_reward": 0.0,
        "waiting_bonus": 0.0,
        "reward_scale": 1 / 48,
        "shaping_scale": SHAPING_SCALE,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": ENT_COEF,
    },
    "normed_waiting": {
        "description": "Normalised + waiting bonus + full_fix params",
        "gamma": 0.99,
        "tiles_away_reward": 0.3,
        "waiting_bonus": 4.0,
        "reward_scale": 1 / 48,
        "shaping_scale": 3.0,
        "defense_weight": DEFENSE_WEIGHT,
        "ent_coef": 0.001,
    },
}


# ---------------------------------------------------------------------------
# Worker (must be top-level for multiprocessing pickling on Windows)
# ---------------------------------------------------------------------------

def _run_variant_worker(args: dict) -> tuple[str, dict]:
    name = args["name"]
    kwargs = {k: v for k, v in args.items() if k != "name"}
    result = train_self_play(**kwargs)
    return name, result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiments(
    names: list[str],
    episodes: int = 5000,
    workers: int | None = None,
    out_dir: Path = Path("models"),
    episodes_per_update: int = 32,
    eval_every: int = 500,
    eval_games: int = 20,
    seed: int = 0,
    resume: bool = False,
) -> dict[str, dict]:
    """
    Train the given variants and return {name: metrics_dict}.

    Runs in parallel when more than one variant is requested and workers > 1.
    In parallel mode verbose output is suppressed per-worker; completion is
    reported as each job finishes.
    """
    out_dir = Path(out_dir)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {list(VARIANTS)}")

    parallel = len(names) > 1 and (workers is None or workers > 1)
    n_workers = min(workers or len(names), len(names))

    jobs = []
    for name in names:
        cfg = VARIANTS[name]
        jobs.append({
            "name": name,
            "n_episodes": episodes,
            "model_path": out_dir / f"{name}.pt",
            "episodes_per_update": episodes_per_update,
            "eval_every": eval_every,
            "eval_games": eval_games,
            "seed": seed,
            "verbose": not parallel,
            "shaping_scale": cfg["shaping_scale"],
            "defense_weight": cfg["defense_weight"],
            "gamma": cfg["gamma"],
            "tiles_away_reward": cfg["tiles_away_reward"],
            "waiting_bonus": cfg["waiting_bonus"],
            "reward_scale": cfg["reward_scale"],
            "ent_coef": cfg["ent_coef"],
            "resume": resume,
        })

    results: dict[str, dict] = {}

    if not parallel:
        for job in jobs:
            name, metrics = _run_variant_worker(job)
            results[name] = metrics
    else:
        print(f"Training {len(jobs)} variants with {n_workers} worker(s) "
              f"(per-step output suppressed)...")
        with multiprocessing.Pool(processes=n_workers) as pool:
            for name, metrics in pool.imap_unordered(_run_variant_worker, jobs):
                print(f"  Done: {name}  best_net={metrics['best_mean_net']:+.3f}  "
                      f"win={metrics['final_win_rate']:.3f}  "
                      f"shoot={metrics['final_shoot_rate']:.3f}")
                results[name] = metrics

    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_table(names: list[str], results: dict[str, dict]) -> None:
    col_w = max(len(n) for n in names) + 2
    desc_w = max(len(VARIANTS[n]["description"]) for n in names) + 2
    sep = "-" * (col_w + desc_w + 42)
    header = (
        f"{'variant':<{col_w}}  {'description':<{desc_w}}"
        f"  {'best_net':>9}  {'win':>6}  {'shoot':>7}  {'draw':>6}"
    )
    print()
    print(header)
    print(sep)
    for name in names:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<{col_w}}  {VARIANTS[name]['description']:<{desc_w}}"
            f"  {r['best_mean_net']:>+9.3f}"
            f"  {r['final_win_rate']:>6.3f}"
            f"  {r['final_shoot_rate']:>7.3f}"
            f"  {r['final_draw_rate']:>6.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train reward-system variants in parallel and compare results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "variants", nargs="*",
        help="Variant names to run (omit to run all). Use --list to see options.",
    )
    parser.add_argument("--list", action="store_true",
                        help="Print available variants and exit.")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Training episodes per variant (default: 5000).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: one per variant).")
    parser.add_argument("--out-dir", type=Path, default=Path("models"),
                        help="Directory for saved model checkpoints (default: models/).")
    parser.add_argument("--batch", type=int, default=32,
                        help="Episodes per PPO update batch (default: 32).")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate every N episodes (default: 500).")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Matches per evaluation (default: 20).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume each variant from its existing checkpoint in --out-dir.")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable variants:")
        col = max(len(n) for n in VARIANTS) + 2
        for name, cfg in VARIANTS.items():
            print(f"  {name:<{col}} {cfg['description']}")
            print(f"  {'':<{col}} gamma={cfg['gamma']}  tiles_away_reward={cfg['tiles_away_reward']}"
                  f"  shaping_scale={cfg['shaping_scale']}  ent_coef={cfg['ent_coef']}")
        print()
        return

    names = args.variants if args.variants else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variant(s): {unknown}. Use --list to see available options.")

    print(f"Running {len(names)} variant(s): {', '.join(names)}")
    if len(names) > 1:
        n_workers = args.workers or len(names)
        print(f"Parallel workers: {n_workers}")
    print()

    results = run_experiments(
        names=names,
        episodes=args.episodes,
        workers=args.workers,
        out_dir=args.out_dir,
        episodes_per_update=args.batch,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        seed=args.seed,
        resume=args.resume,
    )

    _print_table(names, results)


if __name__ == "__main__":
    _cli()
