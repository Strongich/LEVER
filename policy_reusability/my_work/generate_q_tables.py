import pickle

from inference_q import inference_q
import numpy as np
from agents.q_agent import QLearningAgent
from agents.q_agent import SarsaAgent
from DAG import DAG
import time
import pandas as pd

from my_work.init_gridworld import init_gridworld_rand
from utilities import plot_cummulative_reward
from datetime import datetime
import os


def train_q_policy_transitions(grid_world, n_episodes, max_steps_per_episode, agent_type, output_path, learning_rate=None,
                   discount_factor=None, result_step_size=1, plot_cumulative_reward=False):
    # Flatten the grid to get the total number of states
    n_states = np.prod(grid_world.grid.shape)

    # Get the total number of actions
    n_actions = grid_world.action_space.n

    dag = DAG(gridworld=grid_world, N=n_episodes)

    exploration_rate_decay = 0.99999

    # Initialize the Q-Learning agent
    q_agent = None
    if agent_type == "QLearning":
        q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions, exploration_rate_decay=exploration_rate_decay)
    elif agent_type == "Sarsa":
        q_agent = SarsaAgent(n_states=n_states, n_actions=n_actions, exploration_rate_decay=exploration_rate_decay)

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

    transition_history = set()  # {(s, a, r, s1)}

    for episode in range(n_episodes):
        # turn on stopwatch
        start_time = time.time()

        grid_world.reset().flatten()
        state_index = grid_world.state_to_index(grid_world.agent_position)

        for step in range(max_steps_per_episode):
            grid_world.visited_count_states[grid_world.agent_position[0]][grid_world.agent_position[1]] += 1
            action = q_agent.get_action(state_index)

            grid, reward, done, info = grid_world.step(action)

            cumulative_reward += reward
            next_state_index = grid_world.state_to_index(grid_world.agent_position)

            if agent_type == "Sarsa":
                next_action = q_agent.get_action(next_state_index)
                q_agent.update_q_table(state_index, action, reward, next_state_index, next_action)
            elif agent_type == "QLearning":
                q_agent.update_q_table(state_index, action, reward, next_state_index)

            if (state_index != next_state_index):
                transition_history.add((state_index.item(), action, reward, next_state_index.item()))
                dag.add_edge(state_index, next_state_index, reward)

            state_index = next_state_index

            if done:
                break

        # update lerning rate and explortion rate
        q_agent.exploration_rate = max(q_agent.exploration_rate * q_agent.exploration_rate_decay,
                                       q_agent.min_exploration_rate)

        # turn of stopwatch
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # log cumulative reward

        if episode % result_step_size == 0:
            df.at[(episode / result_step_size) + 1, csv_index_episode] = episode
            df.at[(episode / result_step_size) + 1, csv_index_cummulative_reward] = cumulative_reward

    # Save the q_table for future use
    csv_file_name = "Train_" + grid_world.reward_system + "_" + agent_type + "_" + str(n_episodes) + ".csv"
    # df.to_csv(csv_file_name, index=False, header=header)
    if plot_cumulative_reward:
        plot_cummulative_reward(csv_file_name, header[0], header[1])
    np.save(output_path, q_agent.q_table)
    # run.finish()

    return total_time, dag, cumulative_reward, grid_world.visited_count_transitions, transition_history


def main():
    seed = 45

    # Create parent folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"q_tables_seed={seed}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    for reward_system in ("gold", "path", "combined"):
        reward_dir = os.path.join(base_dir, reward_system)
        os.makedirs(reward_dir, exist_ok=True)

        # Create GridWorld instance directly
        grid_world = init_gridworld_rand(seed=seed, reward_system=reward_system)

        print("GridWorld initialized!")
        print(f"Size: {grid_world.grid_width}x{grid_world.grid_length}")
        print(f"Gold positions: {grid_world.gold_positions}")
        print(f"Target position: {grid_world.target_position}")

        # ======== Training parameters ========
        agent_type = "Sarsa"  # or "Q-learning"
        n_episodes = 300000
        max_steps_per_episode = 16 + 16 + 1
        learning_rate = 0.1
        discount_factor = 0.99
        result_step_size = 10
        q_table_output_path = os.path.join(reward_dir, "q_table_example.npy")

        # ======== Train one Q-learning/SARSA agent ========
        total_time, dag, _, _, transition_history = train_q_policy_transitions(
            grid_world,
            n_episodes,
            max_steps_per_episode,
            agent_type,
            q_table_output_path,
            result_step_size=result_step_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        with open(os.path.join(reward_dir, "transition_history.pkl"), "wb") as f:
            pickle.dump(transition_history, f)

        print(f"\nTraining finished in {total_time:.2f}s using {agent_type}")
        print(f"Q-table saved to: {q_table_output_path}")

        # ======== Evaluate trained agent ========
        best_path, cumulative_reward, path = inference_q(
            grid_world=grid_world,
            q_table_path=q_table_output_path
        )

        print(f"\nEvaluation results:")
        print(f"Cumulative reward: {cumulative_reward}")
        print(f"Best path found: {path}")


if __name__ == "__main__":
    main()
