"""Benchmark throughput of the multi-agent reference environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.actions import NO_OP
from src.environments.reference_model_multi_agent import ReferenceModel


def _build_env_config(args: argparse.Namespace) -> dict:
    return {
        "env_name": args.env_name,
        "seed": args.env_seed,
        "deterministic": args.deterministic,
        "num_agents": args.num_agents,
        "steps_per_episode": args.steps_per_episode,
        "sensor_range": args.sensor_range,
        "info_mode": args.info_mode,
        "training_execution_mode": "CTDE",
        "render_env": False,
    }


def _sample_random_actions(env: ReferenceModel, rng: np.random.Generator) -> dict[str, int]:
    return {agent_id: int(rng.integers(0, env.action_space.n)) for agent_id in env.agents}


def _sample_masked_actions(
    env: ReferenceModel,
    obs: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, int]:
    action_mask_slice = env._obs_slices["action_mask"]
    actions = {}
    for agent_id in env.agents:
        mask = obs[agent_id][action_mask_slice]
        valid_actions = np.flatnonzero(mask > 0.5)
        if valid_actions.size == 0:
            actions[agent_id] = NO_OP
        else:
            actions[agent_id] = int(rng.choice(valid_actions))
    return actions


def run_benchmark(
    env_config: dict,
    mode: str,
    steps: int,
    warmup_steps: int,
    action_seed: int,
) -> dict:
    env = ReferenceModel(env_config)
    rng = np.random.default_rng(action_seed)
    obs, _ = env.reset()
    episodes = 0

    def do_step() -> bool:
        nonlocal obs
        if mode == "random":
            actions = _sample_random_actions(env, rng)
        elif mode == "masked":
            actions = _sample_masked_actions(env, obs, rng)
        else:
            msg = f"Unsupported mode: {mode}"
            raise ValueError(msg)

        obs, _rewards, terminated, truncated, _info = env.step(actions)
        done = terminated.get("__all__", False) or truncated.get("__all__", False)
        return done

    for _ in range(warmup_steps):
        if do_step():
            obs, _ = env.reset()

    t0 = time.perf_counter()
    for _ in range(steps):
        done = do_step()
        if done:
            episodes += 1
            obs, _ = env.reset()
    elapsed_s = time.perf_counter() - t0

    return {
        "mode": mode,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "episodes_completed": episodes,
        "elapsed_s": elapsed_s,
        "steps_per_s": steps / elapsed_s,
        "episodes_per_s": episodes / elapsed_s,
        "mean_step_ms": 1000.0 * elapsed_s / steps,
        "env_config": env_config,
    }


def save_results(results: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")
    json_path = output_dir / f"multi_agent_env_benchmark_{timestamp}.json"
    csv_path = output_dir / f"multi_agent_env_benchmark_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)

    fieldnames = [
        "mode",
        "steps",
        "warmup_steps",
        "episodes_completed",
        "elapsed_s",
        "steps_per_s",
        "episodes_per_s",
        "mean_step_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="ReferenceModel-2-1")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--sensor-range", type=int, default=2)
    parser.add_argument("--steps-per-episode", type=int, default=100)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--modes", default="random,masked", help="Comma-separated modes: random, masked")
    parser.add_argument("--env-seed", type=int, default=123)
    parser.add_argument("--action-seed", type=int, default=999)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--info-mode", choices=["lite", "full"], default="lite")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/benchmarks"))
    parser.add_argument(
        "--assert-min-steps-per-s",
        type=float,
        default=None,
        help="If set, assert that the random-mode throughput reaches this threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_config = _build_env_config(args)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    results = []
    for mode in modes:
        result = run_benchmark(
            env_config=env_config,
            mode=mode,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            action_seed=args.action_seed,
        )
        results.append(result)
        print(
            f"[{mode}] steps/s={result['steps_per_s']:.2f}, "
            f"episodes/s={result['episodes_per_s']:.3f}, "
            f"mean_step_ms={result['mean_step_ms']:.3f}"
        )

    json_path, csv_path = save_results(results, args.output_dir)
    print(f"Saved benchmark JSON to {json_path}")
    print(f"Saved benchmark CSV to {csv_path}")

    if args.assert_min_steps_per_s is not None:
        random_result = next((row for row in results if row["mode"] == "random"), None)
        if random_result is None:
            msg = "Assertion requested but random mode is missing from --modes."
            raise ValueError(msg)
        measured = random_result["steps_per_s"]
        if measured < args.assert_min_steps_per_s:
            msg = (
                f"Random-mode throughput {measured:.2f} steps/s is below "
                f"required {args.assert_min_steps_per_s:.2f} steps/s."
            )
            raise AssertionError(msg)


if __name__ == "__main__":
    main()
