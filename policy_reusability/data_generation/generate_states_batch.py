"""
Generate GridWorld trajectories for all configured grid specs and reward systems.
Each seed uses a single layout shared across reward systems.

Configuration is defined in config.py via GRID_SPECS_16 and GRID_SPECS_32.

Defaults (from GRID_SPECS):
- Rewards: path, gold, lever, hazard, hazard-lever, path-gold, path-gold-hazard, path-gold-hazard-lever
- Episodes: varies by spec (X1=30k, X5=150k, X10=300k)
- Max steps per episode: grid_size * grid_size
- Actions: 4 (right, down, right x2, down x2)
- SARSA: alpha=0.1, gamma=0.99, epsilon starts at 1.0, spec-controlled decay, min 0.01
- Snapshot every 1,000 episodes: saves episode_states.npy, episode_actions.npy,
  q_table.npy, dag.pkl; greedy evaluation reward logged to episode_rewards.csv.

Output root per run: state_runs/<spec>_<timestamp>/<reward>/seed_XXXX/...
"""

import argparse
import csv
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import GRID_SPECS_16, GRID_SPECS_32
from policy_reusability.agents.q_agent import SarsaAgent
from policy_reusability.DAG import DAG
from policy_reusability.env.gridworld import GridWorld

# Shared hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def find_rapl_package_paths():
    rapl_root = Path("/sys/class/powercap")
    default = rapl_root / "intel-rapl:0"
    if (default / "energy_uj").exists():
        return default / "energy_uj", default / "max_energy_range_uj"

    candidates = sorted(rapl_root.rglob("energy_uj"))
    if not candidates:
        return None, None

    energy_path = candidates[0]
    max_path = energy_path.parent / "max_energy_range_uj"
    if not max_path.exists():
        return None, None
    return energy_path, max_path


def rapl_delta_uj(prev_uj: int, curr_uj: int, max_range_uj: int) -> int:
    if curr_uj >= prev_uj:
        return curr_uj - prev_uj
    return (max_range_uj - prev_uj) + curr_uj


def sample_layout_for_seed(spec: Dict, seed: int):
    rng = random.Random(seed)
    grid_size = spec["grid_size"]
    agent_initial_position = (0, 0)
    target_position = (grid_size - 1, grid_size - 1)

    total_cells = grid_size * grid_size - 2  # exclude start/target
    hazard_count = spec.get("hazards", max(1, int(total_cells * 0.05)))

    def sample_layout():
        positions = [
            (x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if (x, y) not in [agent_initial_position, target_position]
        ]
        rng.shuffle(positions)
        hazards = positions[:hazard_count]
        gold_start_idx = hazard_count
        golds = positions[gold_start_idx : gold_start_idx + spec["num_golds"]]
        block_start_idx = gold_start_idx + spec["num_golds"]
        blocks = positions[block_start_idx : block_start_idx + spec["num_blocks"]]
        remaining = positions[block_start_idx + spec["num_blocks"] :]
        lever = remaining[0] if remaining else None
        return hazards, golds, blocks, lever

    def has_hazard_free_path(start, goal, hazards, blocks):
        blocked = set(hazards) | set(blocks)
        if start in blocked or goal in blocked:
            return False
        queue = [start]
        visited = set(queue)
        while queue:
            x, y = queue.pop(0)
            if (x, y) == goal:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < grid_size
                    and 0 <= ny < grid_size
                    and (nx, ny) not in blocked
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def layout_is_valid(hazards, blocks, lever):
        if not has_hazard_free_path(
            agent_initial_position, target_position, hazards, blocks
        ):
            return False
        if lever is None:
            return False
        if not has_hazard_free_path(
            agent_initial_position, lever, hazards, blocks
        ):
            return False
        if not has_hazard_free_path(lever, target_position, hazards, blocks):
            return False
        return True

    max_layout_attempts = 200
    for _ in range(max_layout_attempts):
        hazards, golds, blocks, lever = sample_layout()
        if layout_is_valid(hazards, blocks, lever):
            return hazards, golds, blocks, lever
    return sample_layout()


def init_gridworld(
    spec: Dict,
    reward_system: str,
    seed: int,
    layout: tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], tuple[int, int] | None] | None = None,
) -> GridWorld:
    """Create a GridWorld of given size using a shared layout per seed."""
    grid_size = spec["grid_size"]

    agent_initial_position = (0, 0)
    target_position = (grid_size - 1, grid_size - 1)

    if layout is None:
        hazard_positions, gold_positions, block_positions, lever_position = (
            sample_layout_for_seed(spec, seed)
        )
    else:
        hazard_positions, gold_positions, block_positions, lever_position = layout

    gold_positions_list = [list(pos) for pos in gold_positions]
    block_positions_list = [list(pos) for pos in block_positions]
    hazard_positions_list = [list(pos) for pos in hazard_positions]

    hazard_position_value = -2
    lever_position_value = 2
    cell_low_value = min(-1, hazard_position_value)
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -1
    target_reward = +100
    hazard_penalty = 0.0
    step_penalty = 0.0
    exit_without_lever_penalty = 0.0

    grid_world = GridWorld(
        grid_width=grid_size,
        grid_length=grid_size,
        gold_positions=gold_positions_list,
        block_positions=block_positions_list,
        hazard_positions=hazard_positions_list,
        lever_position=list(lever_position) if lever_position else None,
        reward_system=reward_system,
        agent_position=list(agent_initial_position),
        target_position=list(target_position),
        cell_high_value=cell_high_value,
        cell_low_value=cell_low_value,
        start_position_value=start_position_value,
        target_position_value=target_position_value,
        block_position_value=block_position_value,
        gold_position_value=gold_position_value,
        hazard_position_value=hazard_position_value,
        lever_position_value=lever_position_value,
        agent_position_value=agent_position_value,
        block_reward=block_reward,
        target_reward=target_reward,
        gold_k=0,
        n=0,
        action_size=4,
        parameterized=False,
        alpha_beta=(1, 1),
        step_penalty=step_penalty,
        hazard_penalty=hazard_penalty,
        exit_without_lever_penalty=exit_without_lever_penalty,
    )
    return grid_world


def evaluate_greedy(
    spec: Dict,
    q_table: np.ndarray,
    reward_system: str,
    seed: int,
    layout,
) -> float:
    """Greedy rollout (epsilon=0) with the current Q-table on a fresh env."""
    env = init_gridworld(spec, reward_system=reward_system, seed=seed, layout=layout)
    env.reset().flatten()
    state_index = env.state_to_index(env.agent_position)
    total_reward = 0.0
    max_steps = spec["grid_size"] * spec["grid_size"]
    for _ in range(max_steps):
        action = int(np.argmax(q_table[state_index, :]))
        _, reward, done, _ = env.step(action)
        total_reward += reward
        state_index = env.state_to_index(env.agent_position)
        if done:
            break
    return total_reward


def train_seed(
    spec: Dict,
    seed: int,
    reward_system: str,
    output_dir: str,
    layout,
):
    """Train SARSA for one seed/reward and save snapshots."""
    grid_world = init_gridworld(
        spec, reward_system=reward_system, seed=seed, layout=layout
    )

    n_states = np.prod(grid_world.grid.shape)
    n_actions = grid_world.action_space.n

    agent = SarsaAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=ALPHA,
        discount_factor=GAMMA,
        exploration_rate=EPSILON_START,
        exploration_rate_decay=spec.get("epsilon_decay", 0.99999),
        min_exploration_rate=EPSILON_MIN,
    )

    dag = DAG(gridworld=grid_world, N=spec["episodes"])

    episodes_dir = os.path.join(output_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)

    episode_rewards: List[Tuple[int, float]] = []
    episode_times: List[Tuple[int, float]] = []
    episode_energy: List[Tuple[int, float, float]] = []
    width = len(str(spec["episodes"] - 1))  # e.g., 6 digits for 300k
    max_steps = spec["grid_size"] * spec["grid_size"]
    cumulative_time = 0.0
    cumulative_energy_j = 0.0
    interval_energy_j = 0.0
    interval_start_time = time.time()

    energy_path, max_path = find_rapl_package_paths()
    energy_available = energy_path is not None and max_path is not None
    if energy_available:
        max_range_uj = _read_int(max_path)
        prev_energy_uj = _read_int(energy_path)
    else:
        max_range_uj = 0
        prev_energy_uj = 0

    for episode in range(spec["episodes"]):
        episode_start = time.time()
        save_snapshot = episode % spec["save_every"] == 0
        episode_dir = None
        episode_states = None
        episode_actions = None
        episode_hazard_counts = None
        episode_reward = 0.0

        grid_world.reset().flatten()
        if save_snapshot:
            episode_dir = os.path.join(episodes_dir, f"episode_{episode:0{width}d}")
            os.makedirs(episode_dir, exist_ok=True)
            episode_states = [np.copy(grid_world.grid)]
            episode_actions = []
            episode_hazard_counts = [grid_world.hazard_steps]

        state_index = grid_world.state_to_index(grid_world.agent_position)

        for _ in range(max_steps):
            grid_world.visited_count_states[grid_world.agent_position[0]][
                grid_world.agent_position[1]
            ] += 1

            action = agent.get_action(state_index)
            grid, reward, done, _ = grid_world.step(action)

            next_state_index = grid_world.state_to_index(grid_world.agent_position)
            next_action = agent.get_action(next_state_index)

            agent.update_q_table(
                state_index, action, reward, next_state_index, next_action
            )

            if state_index != next_state_index:
                dag.add_edge(state_index, next_state_index)

            if save_snapshot:
                episode_states.append(np.copy(grid))
                episode_actions.append(action)
                episode_hazard_counts.append(grid_world.hazard_steps)

            episode_reward += reward
            state_index = next_state_index

            if done:
                break

        cumulative_time += time.time() - episode_start
        if energy_available:
            curr_energy_uj = _read_int(energy_path)
            delta_uj = rapl_delta_uj(prev_energy_uj, curr_energy_uj, max_range_uj)
            delta_j = delta_uj / 1e6
            cumulative_energy_j += delta_j
            interval_energy_j += delta_j
            prev_energy_uj = curr_energy_uj

        if save_snapshot and episode_dir is not None:
            np.save(
                os.path.join(episode_dir, "episode_states.npy"),
                np.array(episode_states, dtype=np.int8),
            )
            np.save(
                os.path.join(episode_dir, "episode_actions.npy"),
                np.array(episode_actions, dtype=np.int8),
            )
            np.save(
                os.path.join(episode_dir, "episode_hazard_counts.npy"),
                np.array(episode_hazard_counts, dtype=np.int32),
            )
            np.save(os.path.join(episode_dir, "q_table.npy"), agent.q_table)
            with open(os.path.join(episode_dir, "dag.pkl"), "wb") as f:
                pickle.dump(dag, f)

            # Greedy eval on fresh env for this seed/reward
            eval_reward = evaluate_greedy(
                spec, agent.q_table, reward_system, seed, layout
            )
            episode_rewards.append((episode, eval_reward))
            episode_times.append((episode, cumulative_time))
            avg_power_w = 0.0
            if energy_available:
                interval_elapsed = max(time.time() - interval_start_time, 1e-9)
                avg_power_w = interval_energy_j / interval_elapsed
            episode_energy.append((episode, cumulative_energy_j, avg_power_w))
            interval_energy_j = 0.0
            interval_start_time = time.time()

        # Decay epsilon
        agent.exploration_rate = max(
            agent.exploration_rate * agent.exploration_rate_decay,
            agent.min_exploration_rate,
        )

    # Write rewards CSV (snapshot intervals)
    final_reward = evaluate_greedy(spec, agent.q_table, reward_system, seed, layout)
    episode_rewards.append((spec["episodes"], final_reward))
    episode_times.append((spec["episodes"], cumulative_time))
    final_avg_power_w = 0.0
    if energy_available:
        final_elapsed = max(time.time() - interval_start_time, 1e-9)
        final_avg_power_w = interval_energy_j / final_elapsed
    episode_energy.append((spec["episodes"], cumulative_energy_j, final_avg_power_w))

    csv_path = os.path.join(output_dir, "episode_rewards.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "episode", "reward", "time", "energy_j", "avg_power_w"])
        for idx, ((ep, rew), (_, t), (_, e_j, p_w)) in enumerate(
            zip(episode_rewards, episode_times, episode_energy)
        ):
            rounded_rew = int(round(rew))
            writer.writerow([idx, ep, rounded_rew, t, e_j, p_w])

    # Final Q-table for convenience
    np.save(os.path.join(output_dir, "q_table_final.npy"), agent.q_table)
    return cumulative_energy_j


def main():
    parser = argparse.ArgumentParser(
        description="Generate GridWorld trajectories for all reward systems."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="state_runs",
        help="Root directory for generated runs (default: state_runs)",
    )
    parser.add_argument(
        "--spec-set",
        type=str,
        choices=["grid16", "grid32"],
        default="grid16",
        help="Grid spec set to use (default: grid16)",
    )
    args = parser.parse_args()

    if args.spec_set == "grid32":
        grid_specs = GRID_SPECS_32
    else:
        grid_specs = GRID_SPECS_16

    reward_systems = [
        "path",
        "gold",
        "hazard",
        "lever",
        "hazard-lever",
        "path-gold",
        "path-gold-hazard",
        "path-gold-hazard-lever",
    ]

    for spec in grid_specs:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(args.output_root, f"{spec['name']}_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        seeds = list(range(spec["seeds"]))
        layouts = {seed: sample_layout_for_seed(spec, seed) for seed in seeds}
        training_times: Dict[str, float] = {}

        for reward_system in reward_systems:
            print(
                f"=== Training reward system: {reward_system} @ {spec['grid_size']}x{spec['grid_size']} ==="
            )
            reward_start = time.time()
            reward_energy_j = 0.0
            for seed in seeds:
                seed_dir = os.path.join(base_dir, reward_system, f"seed_{seed:04d}")
                os.makedirs(seed_dir, exist_ok=True)

                if os.listdir(seed_dir):
                    print(
                        f"Skipping seed {seed:04d} ({reward_system}); directory not empty."
                    )
                    continue

                print(
                    f"GridWorld initialized ({spec['grid_size']}x{spec['grid_size']}) for seed {seed:04d}, reward {reward_system}"
                )
                seed_energy_j = train_seed(
                    spec=spec,
                    seed=seed,
                    reward_system=reward_system,
                    output_dir=seed_dir,
                    layout=layouts[seed],
                )
                reward_energy_j += seed_energy_j
            reward_elapsed = time.time() - reward_start
            training_times[reward_system] = reward_elapsed
            training_times[f"{reward_system}_energy_j"] = reward_energy_j
            print(
                f"Finished reward {reward_system} in {reward_elapsed / 60:.2f} minutes"
            )

        # Save timing summary
        times_path = os.path.join(base_dir, "training_times.json")
        with open(times_path, "w") as f:
            import json

            json.dump(training_times, f, indent=2)


if __name__ == "__main__":
    main()
