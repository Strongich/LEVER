"""
Generate GridWorld trajectories for multiple grid sizes (default: 16x16) in the
same layout as states_f, training gold, path, and combined for each seed.

Defaults (can be adjusted via GRID_SPECS):
- Grid: 16x16
- Seeds: 20 (seed_0000 ... seed_0019)
- Rewards: gold, path, combined
- Episodes: 300,000
- Max steps per episode: grid_size * 2 + 1 (33 for 16x16)
- Actions: 5 (right, down, right*2, down*2, diagonal)
- SARSA: alpha=0.1, gamma=0.99, epsilon starts at 1.0, decays by 0.99999, min 0.01
- Snapshot every 1,000 episodes: saves episode_states.npy, episode_actions.npy,
  q_table.npy, dag.pkl; greedy evaluation reward logged to episode_rewards.csv.

Output root per grid size: states_<grid_size>/<reward>/<seed>/...
"""

import csv
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from policy_reusability.agents.q_agent import SarsaAgent
from policy_reusability.DAG import DAG
from policy_reusability.env.gridworld import GridWorld

# Define grid configurations; add more entries to generate other sizes.
GRID_SPECS_FIRST = [
    {
        "name": "16",
        "grid_size": 16,
        "num_golds": 50,  # cap at 50 golds
        "num_blocks": 25,
        "episodes": 30_000,
        "save_every": 1_000,
        "seeds": 20,
    },
]
EPSILON_DECAY_OLD = 0.99995


GRID_SPECS = [
    {
        "name": "16",
        "grid_size": 16,
        "num_golds": 50,  # cap at 50 golds
        "num_blocks": 25,
        "episodes": 240_000,
        "save_every": 1_000,
        "seeds": 20,
    },
]

# Shared hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
STEP_PENALTY = 1


def init_gridworld(spec: Dict, reward_system: str, seed: int) -> GridWorld:
    """Create a GridWorld of given size with deterministic layout per seed."""
    random.seed(seed)
    grid_size = spec["grid_size"]

    agent_initial_position = (0, 0)
    target_position = (grid_size - 1, grid_size - 1)

    all_positions = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) not in [agent_initial_position, target_position]
    ]
    random.shuffle(all_positions)
    gold_positions = all_positions[: spec["num_golds"]]
    block_positions = all_positions[
        spec["num_golds"] : spec["num_golds"] + spec["num_blocks"]
    ]

    gold_positions_list = [list(pos) for pos in gold_positions]
    block_positions_list = [list(pos) for pos in block_positions]

    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    grid_world = GridWorld(
        grid_width=grid_size,
        grid_length=grid_size,
        gold_positions=gold_positions_list,
        block_positions=block_positions_list,
        reward_system=reward_system,
        agent_position=list(agent_initial_position),
        target_position=list(target_position),
        cell_high_value=cell_high_value,
        cell_low_value=cell_low_value,
        start_position_value=start_position_value,
        target_position_value=target_position_value,
        block_position_value=block_position_value,
        gold_position_value=gold_position_value,
        agent_position_value=agent_position_value,
        block_reward=block_reward,
        target_reward=target_reward,
        gold_k=0,
        n=0,
        action_size=5,
        parameterized=False,
        alpha_beta=(1, 1),
        step_penalty=STEP_PENALTY,
    )
    return grid_world


def evaluate_greedy(
    spec: Dict, q_table: np.ndarray, reward_system: str, seed: int
) -> float:
    """Greedy rollout (epsilon=0) with the current Q-table on a fresh env."""
    env = init_gridworld(spec, reward_system=reward_system, seed=seed)
    env.reset().flatten()
    state_index = env.state_to_index(env.agent_position)
    total_reward = 0.0
    max_steps = spec["grid_size"] * 2 + 1
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
):
    """Train SARSA for one seed/reward and save snapshots."""
    grid_world = init_gridworld(spec, reward_system=reward_system, seed=seed)

    n_states = np.prod(grid_world.grid.shape)
    n_actions = grid_world.action_space.n

    agent = SarsaAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=ALPHA,
        discount_factor=GAMMA,
        exploration_rate=EPSILON_START,
        exploration_rate_decay=EPSILON_DECAY,
        min_exploration_rate=EPSILON_MIN,
    )

    dag = DAG(gridworld=grid_world, N=spec["episodes"])

    episodes_dir = os.path.join(output_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)

    episode_rewards: List[Tuple[int, float]] = []
    width = len(str(spec["episodes"] - 1))  # e.g., 6 digits for 300k
    max_steps = spec["grid_size"] * 2 + 1

    for episode in range(spec["episodes"]):
        save_snapshot = episode % spec["save_every"] == 0
        episode_dir = None
        episode_states = None
        episode_actions = None
        episode_reward = 0.0

        grid_world.reset().flatten()
        if save_snapshot:
            episode_dir = os.path.join(episodes_dir, f"episode_{episode:0{width}d}")
            os.makedirs(episode_dir, exist_ok=True)
            episode_states = [np.copy(grid_world.grid)]
            episode_actions = []

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

            episode_reward += reward
            state_index = next_state_index

            if done:
                break

        if save_snapshot and episode_dir is not None:
            np.save(
                os.path.join(episode_dir, "episode_states.npy"),
                np.array(episode_states, dtype=np.int8),
            )
            np.save(
                os.path.join(episode_dir, "episode_actions.npy"),
                np.array(episode_actions, dtype=np.int8),
            )
            np.save(os.path.join(episode_dir, "q_table.npy"), agent.q_table)
            with open(os.path.join(episode_dir, "dag.pkl"), "wb") as f:
                pickle.dump(dag, f)

            # Greedy eval on fresh env for this seed/reward
            eval_reward = evaluate_greedy(spec, agent.q_table, reward_system, seed)
            episode_rewards.append((episode, eval_reward))

        # Decay epsilon
        agent.exploration_rate = max(
            agent.exploration_rate * agent.exploration_rate_decay,
            agent.min_exploration_rate,
        )

    # Write rewards CSV (snapshot intervals)
    csv_path = os.path.join(output_dir, "episode_rewards.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "episode", "reward"])
        for idx, (ep, rew) in enumerate(episode_rewards):
            writer.writerow([idx, ep, rew])

    # Final Q-table for convenience
    np.save(os.path.join(output_dir, "q_table_final.npy"), agent.q_table)


def main():
    reward_systems = ["gold", "path", "combined"]

    for spec in GRID_SPECS:
        base_dir = f"states_{spec['name']}"
        os.makedirs(base_dir, exist_ok=True)
        seeds = list(range(spec["seeds"]))
        combined_train_times: List[float] = []

        for reward_system in reward_systems:
            print(
                f"=== Training reward system: {reward_system} @ {spec['grid_size']}x{spec['grid_size']} ==="
            )
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
                start = time.time()
                train_seed(
                    spec=spec,
                    seed=seed,
                    reward_system=reward_system,
                    output_dir=seed_dir,
                )
                elapsed = time.time() - start
                if reward_system == "combined":
                    combined_train_times.append(elapsed)
                print(
                    f"Finished seed {seed:04d} ({reward_system}) in {elapsed / 60:.2f} minutes"
                )
            if reward_system == "combined" and combined_train_times:
                avg_seconds = sum(combined_train_times) / len(combined_train_times)
                print(
                    f"Average combined training time for {len(combined_train_times)} seeds: {avg_seconds / 60:.2f} minutes ({avg_seconds:.1f} seconds)"
                )


if __name__ == "__main__":
    main()
