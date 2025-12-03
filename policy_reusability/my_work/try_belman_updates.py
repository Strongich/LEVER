import pickle
import time

import numpy as np

from my_work.init_gridworld import init_gridworld_rand


def run_bellman_to_convergence(q_table, history, discount_factor,
                               learning_rate, max_iters=1000, tol=1e-6, patience=10):
    """
    Run Bellman updates repeatedly until Q-table converges using a patience criterion.

    Args:
        q_table: np.ndarray [num_states, num_actions]
        history: list of (state, action, reward, next_state)
        discount_factor: gamma
        learning_rate: alpha (used in your bellman_update)
        max_iters: maximum number of iterations
        tol: convergence threshold for max absolute change
        patience: number of consecutive iterations below tol to stop

    Returns:
        np.ndarray: updated Q-table after convergence
    """
    new_q_table = q_table.copy()
    patience_counter = 0

    for i in range(max_iters):
        old_q_table = new_q_table.copy()

        # Perform one Bellman update pass
        new_q_table = bellman_update(
            q_table=old_q_table,
            history=history,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )

        # Compute max absolute change
        max_diff = np.max(np.abs(new_q_table - old_q_table))

        # Check convergence
        if max_diff < tol:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Converged after {i+1} iterations (max change {max_diff:.2e})")
                break
        else:
            patience_counter = 0

    return new_q_table


def bellman_update(q_table, history, learning_rate, discount_factor):
    """
    Perform Bellman Q-learning updates on a Q-table (NumPy array).

    Args:
        q_table: np.ndarray, shape (num_states, num_actions)
        history: list of (state, action, reward, next_state)
        learning_rate: α
        discount_factor: γ
    """
    q_table = q_table.copy()

    for s, a, r, next_s in history:
        # Current Q(s, a)
        q_sa = q_table[s, a]

        # Best possible Q from next state
        max_next_q = np.max(q_table[next_s])

        # Bellman update
        q_table[s, a] = q_sa + learning_rate * (r + discount_factor * max_next_q - q_sa)

    return q_table


def merge_transition_histories(history1, history2):
    """
    Merge two transition histories by summing rewards for common (state, action) pairs.
    If next_states differ, keep the one with the higher total reward.

    Each history is a list of tuples: (state, action, reward, next_state)
    """
    # Convert lists to dicts for quick lookup
    dict1 = {(s, a): (r, ns) for s, a, r, ns in history1}
    dict2 = {(s, a): (r, ns) for s, a, r, ns in history2}

    # Find intersection of (state, action) pairs
    common_keys = set(dict1.keys()) & set(dict2.keys())

    merged_history = []
    for key in common_keys:
        (r1, ns1), (r2, ns2) = dict1[key], dict2[key]
        total_reward = r1 + r2

        if ns1 == ns2:
            merged_history.append((key[0], key[1], total_reward, ns1))
        else:
            # Keep the next_state associated with the higher reward
            if r1 >= r2:
                merged_history.append((key[0], key[1], total_reward, ns1))
            else:
                merged_history.append((key[0], key[1], total_reward, ns2))

    return merged_history


def inference_q(grid_world, q_table):

    # run = wandb.init(project="Inference_Q")
    total_time = 0
    total_reward = 0

    # Reset the environment to its initial state
    grid_world.reset().flatten()
    state_index = grid_world.state_to_index(grid_world.agent_position)

    # Maximum number of steps for inference
    max_steps_inference = 100
    path = []

    for step in range(max_steps_inference):
        # turn on stopwatch
        start_time = time.time()

        # greedy action selection (inference)
        action = np.argmax(q_table[state_index, :])
        path.append(action)

        # step
        grid, reward, done, _ = grid_world.step(action)
        total_reward += reward
        next_state_index = grid_world.state_to_index(grid_world.agent_position)

        # upadate state index
        state_index = next_state_index

        # check if the agent reached the target or the maximum number of steps is reached
        if done:
            if reward > 0:
                print("Agent reached the target!")
            else:
                print("Agent failed to reach the target!")
            break

        # turn of stopwatch
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # wandb.log({"Total Inference Time": total_time}, step=step)

    # run.finish()
    return total_time, total_reward, path


def main():
    base_dir = "q_tables_seed=44_20251108_014918"
    seed = 44

    transition_histories = {}
    q_tables = {}

    for reward in ("gold", "path", "combined"):
        tr_path = f"{base_dir}/{reward}/transition_history.pkl"
        with open(tr_path, "rb") as f:
            transition_histories[reward] = pickle.load(f)

        q_tables[reward] = np.load(f"{base_dir}/{reward}/q_table_example.npy")

    merged_tr_history = merge_transition_histories(
        transition_histories['gold'],
        transition_histories['path']
    )

    # new_q_table = q_tables['gold'] + q_tables['path']
    new_q_table = np.zeros_like(q_tables['gold'])
    new_q_table = run_bellman_to_convergence(
        q_table=new_q_table,
        history=merged_tr_history,
        discount_factor=0.99,
        learning_rate=1.0,
        max_iters=7000,
        tol=1e-6,
        patience=10
    )

    grid_world = init_gridworld_rand("combined", seed=seed)

    best_path1, cumulative_reward1, path1 = inference_q(
        grid_world=grid_world,
        q_table=q_tables['combined']
    )

    best_path2, cumulative_reward2, path2 = inference_q(
        grid_world=grid_world,
        q_table=new_q_table
    )

    print(f"Cumulative reward from training on combined environment: {cumulative_reward1}")
    print(f"Cumulative reward from Bellman updates: {cumulative_reward2}")


if __name__ == '__main__':
    main()