import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from policy_reusability.DAG import DAG
from policy_reusability.agents.q_agent import QLearningAgent, SarsaAgent
from policy_reusability.data_generation.gridworld_factory import init_gridworld_rand
from policy_reusability.inference_q import inference_q
from policy_reusability.utilities import plot_cummulative_reward
from pathlib import Path


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


def train_q_policy_for_states(
    grid_world,
    n_episodes,
    max_steps_per_episode,
    agent_type,
    output_path,
    learning_rate=None,
    discount_factor=None,
    result_step_size=1,
    plot_cumulative_reward=False,
    directory_to_save=None,
):
    # Flatten the grid to get the total number of states
    n_states = np.prod(grid_world.grid.shape)

    # Get the total number of actions
    n_actions = grid_world.action_space.n

    dag = DAG(gridworld=grid_world, N=n_episodes)

    exploration_rate_decay = 0.99999

    # Initialize the Q-Learning agent
    q_agent = None
    if agent_type == "QLearning":
        q_agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            exploration_rate_decay=exploration_rate_decay,
        )
    elif agent_type == "Sarsa":
        q_agent = SarsaAgent(
            n_states=n_states,
            n_actions=n_actions,
            exploration_rate_decay=exploration_rate_decay,
        )

    # check if we want to hardcode lr and df by using input parameters
    if learning_rate != None:
        q_agent.learning_rate = learning_rate
    if discount_factor != None:
        q_agent.discount_factor = discount_factor

    df = pd.DataFrame()
    csv_index_episode = 0
    csv_index_cummulative_reward = 1

    header = ["Episode", "Cumulative Reward"]
    cumulative_reward = 0
    total_time = 0

    episodes_to_save = []
    episode_rewards = []
    episode_times = []
    episode_energy = []
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

    episodes_dir = os.path.join(directory_to_save, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)

    width = len(str(n_episodes - 1))  # number of digits needed
    for episode in range(n_episodes):
        is_episode_to_save = episode % 1000 == 0
        episode_dir = None
        if is_episode_to_save:
            episodes_to_save.append(episode)
            episode_dir = os.path.join(episodes_dir, f"episode_{episode:0{width}d}")
            os.makedirs(episode_dir, exist_ok=True)

        # turn on stopwatch
        start_time = time.time()

        grid_world.reset().flatten()
        state_index = grid_world.state_to_index(grid_world.agent_position)

        episode_states = None
        episode_actions = None
        episode_reward = None
        episode_hazard_counts = None
        if is_episode_to_save:
            episode_states = [np.copy(grid_world.grid)]
            episode_actions = []

            episode_reward = 0
            episode_hazard_counts = [grid_world.hazard_steps]

        for step in range(max_steps_per_episode):
            grid_world.visited_count_states[grid_world.agent_position[0]][
                grid_world.agent_position[1]
            ] += 1
            action = q_agent.get_action(state_index)

            grid, reward, done, info = grid_world.step(action)

            if is_episode_to_save:
                episode_states.append(np.copy(grid))
                episode_actions.append(action)

                episode_reward += reward
                episode_hazard_counts.append(grid_world.hazard_steps)

            cumulative_reward += reward
            next_state_index = grid_world.state_to_index(grid_world.agent_position)

            if agent_type == "Sarsa":
                next_action = q_agent.get_action(next_state_index)
                q_agent.update_q_table(
                    state_index, action, reward, next_state_index, next_action
                )
            elif agent_type == "QLearning":
                q_agent.update_q_table(state_index, action, reward, next_state_index)

            if state_index != next_state_index:
                dag.add_edge(state_index, next_state_index)
            state_index = next_state_index

            if done:
                break

        if is_episode_to_save:
            episode_rewards.append(episode_reward)
            episode_times.append(cumulative_time)
            avg_power_w = 0.0
            if energy_available:
                interval_elapsed = max(time.time() - interval_start_time, 1e-9)
                avg_power_w = interval_energy_j / interval_elapsed
            episode_energy.append((cumulative_energy_j, avg_power_w))
            interval_energy_j = 0.0
            interval_start_time = time.time()

            episode_states = np.array(episode_states, dtype=np.int8)
            episode_actions = np.array(episode_actions, dtype=np.int8)
            episode_hazard_counts = np.array(episode_hazard_counts, dtype=np.int32)
            np.save(os.path.join(episode_dir, "episode_states.npy"), episode_states)
            np.save(os.path.join(episode_dir, "episode_actions.npy"), episode_actions)
            np.save(
                os.path.join(episode_dir, "episode_hazard_counts.npy"),
                episode_hazard_counts,
            )
            np.save(os.path.join(episode_dir, "q_table.npy"), q_agent.q_table)

        # update lerning rate and explortion rate
        q_agent.exploration_rate = max(
            q_agent.exploration_rate * q_agent.exploration_rate_decay,
            q_agent.min_exploration_rate,
        )

        # turn of stopwatch
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        cumulative_time += elapsed_time
        if energy_available:
            curr_energy_uj = _read_int(energy_path)
            delta_uj = rapl_delta_uj(prev_energy_uj, curr_energy_uj, max_range_uj)
            delta_j = delta_uj / 1e6
            cumulative_energy_j += delta_j
            interval_energy_j += delta_j
            prev_energy_uj = curr_energy_uj

        # log cumulative reward

        if episode % result_step_size == 0:
            df.at[(episode / result_step_size) + 1, csv_index_episode] = episode
            df.at[(episode / result_step_size) + 1, csv_index_cummulative_reward] = (
                cumulative_reward
            )

    episode_rewards_df = pd.DataFrame(
        {
            "episode": episodes_to_save,
            "reward": [int(round(r)) for r in episode_rewards],
            "time": episode_times,
            "energy_j": [e for e, _ in episode_energy],
            "avg_power_w": [p for _, p in episode_energy],
        }
    )
    episode_rewards_df.to_csv(os.path.join(directory_to_save, "episode_rewards.csv"))

    # Save the q_table for future use
    csv_file_name = (
        "Train_"
        + grid_world.reward_system
        + "_"
        + agent_type
        + "_"
        + str(n_episodes)
        + ".csv"
    )
    df.to_csv(csv_file_name, index=False, header=header)
    if plot_cumulative_reward:
        plot_cummulative_reward(csv_file_name, header[0], header[1])
    np.save(output_path, q_agent.q_table)
    # run.finish()

    return total_time, dag, cumulative_reward, grid_world.visited_count_transitions


def main():
    # Create parent folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"states_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    for reward_system in ("path",):
        reward_dir = os.path.join(base_dir, reward_system)
        os.makedirs(reward_dir, exist_ok=True)

        for seed in range(114, 200):
            seed_dir = os.path.join(reward_dir, f"seed_{seed:04d}")
            os.makedirs(seed_dir, exist_ok=True)

            # Create GridWorld instance directly
            grid_world = init_gridworld_rand(seed=seed, reward_system=reward_system)

            print("GridWorld initialized!")
            print(f"Size: {grid_world.grid_width}x{grid_world.grid_length}")
            print(f"Gold positions: {grid_world.gold_positions}")
            print(f"Target position: {grid_world.target_position}")

            # ======== Training parameters ========
            agent_type = "Sarsa"  # or "Q-learning"
            n_episodes = 300000
            max_steps_per_episode = grid_world.grid_width * grid_world.grid_length
            learning_rate = 0.1
            discount_factor = 0.99
            result_step_size = 10
            q_table_output_path = "q_table_example.npy"

            # ======== Train one Q-learning/SARSA agent ========
            total_time, dag, _, _ = train_q_policy_for_states(
                grid_world,
                n_episodes,
                max_steps_per_episode,
                agent_type,
                q_table_output_path,
                result_step_size=result_step_size,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                directory_to_save=seed_dir,
            )

            print(f"\nTraining finished in {total_time:.2f}s using {agent_type}")
            print(f"Q-table saved to: {q_table_output_path}")

            # ======== Evaluate trained agent ========
            best_path, cumulative_reward, path = inference_q(
                grid_world=grid_world, q_table_path=q_table_output_path
            )

            print("\nEvaluation results:")
            print(f"Cumulative reward: {cumulative_reward}")
            print(f"Best path found: {path}")


if __name__ == "__main__":
    main()
