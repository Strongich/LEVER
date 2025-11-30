import glob
import os
import pickle
import random

import numpy as np
import pandas as pd

# GridWorld value constants
START_POSITION_VALUE = 5
TARGET_POSITION_VALUE = 10
BLOCK_POSITION_VALUE = -1
GOLD_POSITION_VALUE = 1
AGENT_POSITION_VALUE = 7

# Feature vector constants
STATE_HEAD_LEN = 13  # First 13 scalars before the variable manhattan list
GMAX = 50  # Maximum number of golds to pad manhattan distances to


def state_to_vector(state: np.ndarray) -> np.ndarray:
    """
    Convert a GridWorld state (16x16 grid) into a fixed-size feature vector.

    The feature vector contains (in order):
    1. agent_xy: (x, y) normalized by N
    2. nearest_gold_vec: (dx, dy) normalized by N
    3. nearest_gold_dist: d_near normalized by sqrt(2)*N
    4. num_golds_remaining: count (unnormalized)
    5. exit_vec: (dx, dy) to target normalized by N
    6. exit_dist: distance to target normalized by sqrt(2)*N
    7. walls_nearby: [up, down, left, right] binary indicators (unnormalized)
    8. manhattan_to_golds: Manhattan distances normalized by 2*N (sorted, padded/truncated to GMAX)

    Args:
        state: numpy array of shape (16, 16) representing the GridWorld state

    Returns:
        numpy array of shape (STATE_HEAD_LEN + GMAX,) = (63,) containing the flattened feature vector

    Raises:
        ValueError: If no agent or gold found in the state
        Note: If exit (value 10) is not found, it is assumed the agent is at the exit
              position (terminal state), and exit-related features are set to zero.
    """
    if state.shape != (16, 16):
        raise ValueError(f"Expected state shape (16, 16), got {state.shape}")

    N = 16  # Grid size

    # Find agent position (x, y)
    agent_positions = np.argwhere(state == AGENT_POSITION_VALUE)
    if len(agent_positions) == 0:
        raise ValueError("No agent found in state (value 7)")
    agent_y, agent_x = agent_positions[0]  # Note: argwhere returns (row, col)

    # Find target/exit position
    # If exit is not found, agent is likely on the exit (terminal state)
    # In this case, use agent position as exit position (exit vector/distance will be zero)
    exit_positions = np.argwhere(state == TARGET_POSITION_VALUE)
    if len(exit_positions) == 0:
        # Agent is at exit position - use agent position as exit
        exit_y, exit_x = agent_y, agent_x
    else:
        exit_y, exit_x = exit_positions[0]

    # Find all gold positions
    gold_positions = np.argwhere(state == GOLD_POSITION_VALUE)
    if len(gold_positions) == 0:
        raise ValueError("No gold found in state (value 1)")

    # Number of golds remaining
    num_golds_remaining = len(gold_positions)

    # Convert gold positions from (row, col) to (x, y) coordinates
    gold_coords = [(col, row) for row, col in gold_positions]

    # Compute distances to all gold positions
    manhattan_distances = []
    euclidean_distances = []
    vectors_to_gold = []

    for gold_x, gold_y in gold_coords:
        dx = gold_x - agent_x
        dy = gold_y - agent_y

        vectors_to_gold.append((dx, dy))

        # Manhattan distance: d_1 = |x - g_x| + |y - g_y|
        d_manhattan = abs(dx) + abs(dy)
        manhattan_distances.append(d_manhattan)

        # Euclidean distance
        d_euclidean = np.sqrt(dx**2 + dy**2)
        euclidean_distances.append(d_euclidean)

    # Find nearest gold
    nearest_idx = np.argmin(euclidean_distances)
    nearest_dx, nearest_dy = vectors_to_gold[nearest_idx]
    d_near = euclidean_distances[nearest_idx]

    # Compute exit/target vector and distance
    exit_dx = exit_x - agent_x
    exit_dy = exit_y - agent_y
    exit_dist = np.sqrt(exit_dx**2 + exit_dy**2)

    # Check walls in 4 directions: up, down, left, right
    # up: y-1, down: y+1, left: x-1, right: x+1
    wall_up = (
        1
        if (agent_y - 1 < 0 or state[agent_y - 1, agent_x] == BLOCK_POSITION_VALUE)
        else 0
    )
    wall_down = (
        1
        if (agent_y + 1 >= N or state[agent_y + 1, agent_x] == BLOCK_POSITION_VALUE)
        else 0
    )
    wall_left = (
        1
        if (agent_x - 1 < 0 or state[agent_y, agent_x - 1] == BLOCK_POSITION_VALUE)
        else 0
    )
    wall_right = (
        1
        if (agent_x + 1 >= N or state[agent_y, agent_x + 1] == BLOCK_POSITION_VALUE)
        else 0
    )

    # Normalize continuous features
    x_tilde = agent_x / N
    y_tilde = agent_y / N
    dx_tilde = nearest_dx / N
    dy_tilde = nearest_dy / N
    d_near_tilde = d_near / (np.sqrt(2) * N)
    exit_dx_tilde = exit_dx / N
    exit_dy_tilde = exit_dy / N
    exit_dist_tilde = exit_dist / (np.sqrt(2) * N)

    # Normalize Manhattan distances: d_1_tilde = d_1 / (2*N)
    manhattan_distances_tilde = np.array(
        [d / (2 * N) for d in manhattan_distances], dtype=np.float32
    )
    manhattan_distances_tilde.sort()

    # Pad/truncate manhattan distances to GMAX
    k = len(manhattan_distances_tilde)
    d1_fixed = np.zeros(GMAX, dtype=np.float32)
    take = min(k, GMAX)
    if take > 0:
        d1_fixed[:take] = manhattan_distances_tilde[:take]

    # Construct feature vector head (first 13 scalars)
    head = np.array(
        [
            # agent_xy
            x_tilde,
            y_tilde,
            # nearest_gold_vec
            dx_tilde,
            dy_tilde,
            # nearest_gold_dist
            d_near_tilde,
            # num_golds_remaining
            num_golds_remaining,
            # exit_vec
            exit_dx_tilde,
            exit_dy_tilde,
            # exit_dist
            exit_dist_tilde,
            # walls_nearby [up, down, left, right]
            wall_up,
            wall_down,
            wall_left,
            wall_right,
        ],
        dtype=np.float32,
    )

    # Concatenate head and padded manhattan distances
    feature_vector = np.concatenate([head, d1_fixed], axis=0)

    return feature_vector


def get_episode_path(base_dir, policy, seed, episode_id):
    # episode_id is an integer, need to pad to 6 digits
    episode_str = f"episode_{int(episode_id):06d}"
    return os.path.join(
        base_dir,
        "states_f",
        policy,
        seed,
        "episodes",
        episode_str,
        "episode_states.npy",
    )


def create_canonical_states():
    """
    Randomly select 64 states (32 from gold, 32 from path), convert them to feature vectors
    using state_to_vector(), and save them as data/canonical_states.npy.
    Only creates the file if it doesn't already exist.

    Returns:
        numpy array of shape (64, STATE_HEAD_LEN + GMAX) = (64, 63) containing the canonical
        state feature vectors, or None if file already exists
    """
    base_dir = os.getcwd()
    output_path = os.path.join(base_dir, "data", "canonical_states.npy")

    # Check if file already exists
    if os.path.exists(output_path):
        print(
            f"canonical_states.npy already exists at {output_path}. Skipping creation."
        )
        return np.load(output_path)

    # Find all episode_states.npy files for each policy
    gold_pattern = os.path.join(
        base_dir, "states_f", "gold", "**", "episode_states.npy"
    )
    path_pattern = os.path.join(
        base_dir, "states_f", "path", "**", "episode_states.npy"
    )

    gold_files = glob.glob(gold_pattern, recursive=True)
    path_files = glob.glob(path_pattern, recursive=True)

    print(
        f"Found {len(gold_files)} gold episode files and {len(path_files)} path episode files"
    )

    # Randomly select episodes and then randomly select states from those episodes
    gold_states = []
    path_states = []

    # Randomly shuffle episode files
    random.shuffle(gold_files)
    random.shuffle(path_files)

    # Collect 32 states from gold episodes
    for npy_path in gold_files:
        if len(gold_states) >= 32:
            break
        try:
            states = np.load(npy_path)
            if states.ndim == 3:
                # Randomly select one state from this episode
                if len(states) > 0:
                    random_idx = random.randint(0, len(states) - 1)
                    state = states[random_idx]
                    try:
                        feature_vec = state_to_vector(state)
                        gold_states.append(feature_vec)
                    except Exception as e:
                        print(f"Error converting state to vector from {npy_path}: {e}")
                        continue
            elif states.ndim == 2:
                # Single state
                try:
                    feature_vec = state_to_vector(states)
                    gold_states.append(feature_vec)
                except Exception as e:
                    print(f"Error converting state to vector from {npy_path}: {e}")
                    continue
            else:
                continue
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue

    # Collect 32 states from path episodes
    for npy_path in path_files:
        if len(path_states) >= 32:
            break
        try:
            states = np.load(npy_path)
            if states.ndim == 3:
                # Randomly select one state from this episode
                if len(states) > 0:
                    random_idx = random.randint(0, len(states) - 1)
                    state = states[random_idx]
                    try:
                        feature_vec = state_to_vector(state)
                        path_states.append(feature_vec)
                    except Exception as e:
                        print(f"Error converting state to vector from {npy_path}: {e}")
                        continue
            elif states.ndim == 2:
                # Single state
                try:
                    feature_vec = state_to_vector(states)
                    path_states.append(feature_vec)
                except Exception as e:
                    print(f"Error converting state to vector from {npy_path}: {e}")
                    continue
            else:
                continue
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue

    # Check if we have enough states
    if len(gold_states) < 32:
        print(f"Warning: Only collected {len(gold_states)} gold states (need 32)")
    if len(path_states) < 32:
        print(f"Warning: Only collected {len(path_states)} path states (need 32)")

    # Combine into single array
    canonical_states = np.array(gold_states + path_states, dtype=np.float32)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    np.save(output_path, canonical_states)
    print(f"Saved {len(canonical_states)} canonical states to {output_path}")

    return canonical_states


def process_states():
    """
    Process states from gold and path policies and save to data/processed_states.csv.
    Transitions are stored in memory as tuples of numpy arrays (np.array, np.array).
    If the file already exists, returns it as a pandas DataFrame without reprocessing.

    Returns:
        pandas DataFrame with columns: policy_target, policy_name, reward, transitions
        where transitions is a list of tuples (np.array, np.array)
    """
    base_dir = os.getcwd()
    output_path = os.path.join(base_dir, "data", "processed_states.csv")
    transitions_path = os.path.join(
        base_dir, "data", "processed_states_transitions.pkl"
    )

    # Check if file already exists
    if os.path.exists(output_path) and os.path.exists(transitions_path):
        print(
            f"processed_states.csv already exists at {output_path}. Loading and returning."
        )
        df = pd.read_csv(output_path)
        # Load transitions from pickle file
        with open(transitions_path, "rb") as f:
            transitions_list = pickle.load(f)
        df["transitions"] = transitions_list
        return df

    policies = ["gold", "path"]
    percentages = [0.2, 0.6, 1.0]

    results = []

    # First, find seeds that exist in BOTH policies
    gold_seed_pattern = os.path.join(base_dir, "states_f", "gold", "seed_*")
    path_seed_pattern = os.path.join(base_dir, "states_f", "path", "seed_*")

    gold_seed_dirs = glob.glob(gold_seed_pattern)
    path_seed_dirs = glob.glob(path_seed_pattern)

    # Extract seed names (basename) from directories
    gold_seed_names = {os.path.basename(d) for d in gold_seed_dirs}
    path_seed_names = {os.path.basename(d) for d in path_seed_dirs}

    # Find common seeds (seeds that exist in both policies)
    common_seed_names = gold_seed_names.intersection(path_seed_names)
    common_seed_names = sorted(list(common_seed_names))

    print(
        f"Found {len(gold_seed_names)} gold seeds and {len(path_seed_names)} path seeds"
    )
    print(f"Found {len(common_seed_names)} common seeds")

    # Select only 10% of common seeds
    num_seeds_to_use = max(1, int(len(common_seed_names) * 0.1))
    selected_seed_names = random.sample(common_seed_names, num_seeds_to_use)

    print(
        f"Using {len(selected_seed_names)} common seeds (10% of {len(common_seed_names)})"
    )
    print(f"Selected seeds: {selected_seed_names}")

    # Now process both policies using the same selected seeds
    for policy in policies:
        print(f"\nProcessing policy: {policy}")

        for seed_name in selected_seed_names:
            seed_dir = os.path.join(base_dir, "states_f", policy, seed_name)
            rewards_file = os.path.join(seed_dir, "episode_rewards.csv")

            if not os.path.exists(rewards_file):
                print(f"Warning: {rewards_file} not found. Skipping.")
                continue

            try:
                df_rewards = pd.read_csv(rewards_file)
            except Exception as e:
                print(f"Error reading {rewards_file}: {e}")
                continue

            total_episodes = len(df_rewards)
            if total_episodes == 0:
                continue

            for p in percentages:
                # Calculate index
                # 100% -> last index (N-1)
                # 20% -> 0.2 * (N-1)
                idx = int(p * (total_episodes - 1))

                if idx >= total_episodes:
                    idx = total_episodes - 1

                row = df_rewards.iloc[idx]
                episode_id = row["episode"]
                reward = row["reward"]

                npy_path = get_episode_path(base_dir, policy, seed_name, episode_id)

                if not os.path.exists(npy_path):
                    print(f"Warning: {npy_path} not found. Skipping.")
                    continue

                try:
                    states = np.load(npy_path)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue

                # Process states
                state_vectors = []
                for state in states:
                    try:
                        vec = state_to_vector(state)
                        state_vectors.append(vec)
                    except Exception as e:
                        print(f"Error processing state in {npy_path}: {e}")
                        # If one state fails, maybe we should skip the episode or continue?
                        # Assuming robust pi2vec, but let's handle gracefully
                        continue

                # Create pairs (s_t, s_{t+1})
                pairs = []
                for i in range(len(state_vectors) - 1):
                    pairs.append((state_vectors[i], state_vectors[i + 1]))

                # Format policy name
                # policy_name: {target}_{seed_number}_{20 or 40 or 60 or 80}
                # User asked for 20, 40, 60, 100
                policy_name = (
                    f"{policy}_{seed_name.replace('seed_', '')}_{int(p * 100)}"
                )

                results.append(
                    {
                        "policy_target": policy,
                        "policy_name": policy_name,
                        "reward": reward,
                        "transitions": pairs,
                        "episode_id": episode_id,
                        "seed_name": seed_name,
                    }
                )

    # Extract transitions before creating DataFrame (to keep them as tuples of numpy arrays)
    transitions_list = [result["transitions"] for result in results]

    # Create DataFrame without transitions column (for CSV storage)
    df_metadata = pd.DataFrame(
        [
            {
                "policy_target": result["policy_target"],
                "policy_name": result["policy_name"],
                "reward": result["reward"],
                "episode_id": result["episode_id"],
                "seed_name": result["seed_name"],
            }
            for result in results
        ]
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save metadata to CSV
    df_metadata.to_csv(output_path, index=False)

    # Save transitions separately as pickle (to preserve numpy arrays)
    with open(transitions_path, "wb") as f:
        pickle.dump(transitions_list, f)

    print(f"Saved processed states metadata to {output_path}")
    print(f"Saved transitions to {transitions_path}")

    # Add transitions back to DataFrame for return (as tuples of numpy arrays)
    df_metadata["transitions"] = transitions_list

    return df_metadata
