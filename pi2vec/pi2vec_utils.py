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
HAZARD_POSITION_VALUE = -2
LEVER_POSITION_VALUE = 2
AGENT_POSITION_VALUE = 7

# Feature vector constants
STATE_HEAD_LEN = 21  # First scalars before the variable manhattan list
GMAX_16 = 50  # Maximum number of golds to pad manhattan distances to
GMAX_64 = 200  # Maximum number of golds to pad manhattan distances to


def state_to_vector(state: np.ndarray, hazard_steps: float | None = None) -> np.ndarray:
    """
    Convert a GridWorld state (NxN grid) into a fixed-size feature vector.

    The feature vector contains (in order):
    1. agent_xy: (x, y) normalized by N
    2. nearest_gold_vec: (dx, dy) normalized by N
    3. nearest_gold_dist: d_near normalized by sqrt(2)*N
    4. num_golds_remaining: count (unnormalized)
    5. exit_vec: (dx, dy) to target normalized by N
    6. exit_dist: distance to target normalized by sqrt(2)*N
    7. walls_nearby: [up, down, left, right] binary indicators (unnormalized)
    8. lever_vec: (dx, dy) to lever normalized by N (zero if no lever present)
    9. lever_dist: distance to lever normalized by sqrt(2)*N (zero if no lever present)
    10. lever_collected_flag: 1 if lever value absent in grid, else 0
    11. hazard_count: hazards / N^2 (normalized density)
    12. nearest_hazard_dist: min distance to a hazard normalized by sqrt(2)*N (zero if none)
    13. hazard_steps: cumulative hazard visits so far (normalized by N^2)
    14. manhattan_to_golds: Manhattan distances normalized by 2*N (sorted, padded/truncated to GMAX)

    Args:
        state: numpy array of shape (N, N) representing the GridWorld state (e.g., (16, 16) or (64, 64))

    Returns:
        numpy array of shape (STATE_HEAD_LEN + GMAX,) containing the flattened feature vector
        where GMAX depends on grid size (GMAX_16=50 for 16x16, GMAX_64=200 for 64x64)

    Raises:
        ValueError: If state is not square, or if no agent or gold found in the state
        Note: If exit (value 10) is not found, it is assumed the agent is at the exit
              position (terminal state), and exit-related features are set to zero.
    """
    # Check if state is square
    if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
        raise ValueError(f"Expected square state shape (N, N), got {state.shape}")

    N = state.shape[0]  # Grid size (inferred from state shape)

    # Determine GMAX based on grid size
    if N == 16:
        GMAX = GMAX_16
    elif N == 64:
        GMAX = GMAX_64
    else:
        # Default to GMAX_16 for other sizes, but could be made configurable
        GMAX = GMAX_16

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
        # Robust fallback: no gold in state; zero out gold-dependent features
        num_golds_remaining = 0
        nearest_dx = nearest_dy = d_near = 0.0
        manhattan_distances = []
    else:
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

    # Hazards
    hazard_positions = np.argwhere(state == HAZARD_POSITION_VALUE)
    hazard_count = len(hazard_positions)
    hazard_density = hazard_count / float(N * N)
    if hazard_count == 0:
        nearest_hazard_dist = 0.0
    else:
        h_euclidean = []
        for hy, hx in hazard_positions:
            dx = hx - agent_x
            dy = hy - agent_y
            h_euclidean.append(np.sqrt(dx**2 + dy**2))
        nearest_hazard_dist = float(np.min(h_euclidean))

    # Lever
    lever_positions = np.argwhere(state == LEVER_POSITION_VALUE)
    lever_present = len(lever_positions) > 0
    if lever_present:
        lever_y, lever_x = lever_positions[0]
        lever_dx = lever_x - agent_x
        lever_dy = lever_y - agent_y
        lever_dist = np.sqrt(lever_dx**2 + lever_dy**2)
    else:
        lever_dx = lever_dy = lever_dist = 0.0

    # Heuristic flag: if lever cell is absent, assume it has been collected
    lever_collected_flag = 0 if lever_present else 1

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
    lever_dx_tilde = lever_dx / N
    lever_dy_tilde = lever_dy / N
    lever_dist_tilde = lever_dist / (np.sqrt(2) * N)
    nearest_hazard_dist_tilde = nearest_hazard_dist / (np.sqrt(2) * N)
    hazard_steps_norm = 0.0
    if hazard_steps is not None:
        hazard_steps_norm = hazard_steps / float(N * N)

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

    # Construct feature vector head
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
            # lever vector/dist
            lever_dx_tilde,
            lever_dy_tilde,
            lever_dist_tilde,
            # lever collected flag (heuristic)
            lever_collected_flag,
            # hazard features
            hazard_density,
            nearest_hazard_dist_tilde,
            hazard_steps_norm,
        ],
        dtype=np.float32,
    )

    # Concatenate head and padded manhattan distances
    feature_vector = np.concatenate([head, d1_fixed], axis=0)

    return feature_vector


def get_episode_path(
    base_dir, policy, seed, episode_id, states_folder: str | None = "states_16"
):
    episode_id_int = int(episode_id)
    if states_folder:
        root_dir = os.path.join(base_dir, states_folder)
    else:
        root_dir = base_dir
    episodes_dir = os.path.join(root_dir, policy, seed, "episodes")
    # Try common zero-padding widths first.
    for width in (6, 5, 4):
        episode_str = f"episode_{episode_id_int:0{width}d}"
        candidate = os.path.join(episodes_dir, episode_str, "episode_states.npy")
        if os.path.exists(candidate):
            return candidate
    # Fallback: scan existing episode folders and match numeric suffix.
    if os.path.isdir(episodes_dir):
        for name in os.listdir(episodes_dir):
            if not name.startswith("episode_"):
                continue
            suffix = name.split("_", 1)[1]
            if suffix.isdigit() and int(suffix) == episode_id_int:
                return os.path.join(episodes_dir, name, "episode_states.npy")
    # Default to 6-digit layout if nothing matches.
    episode_str = f"episode_{episode_id_int:06d}"
    return os.path.join(episodes_dir, episode_str, "episode_states.npy")


def create_canonical_states(
    states_folder: str = "states_16",
    canonical_states: int = 64,
    run_dir: str | None = None,
    reward_systems: list[str] | None = None,
    force_recreate: bool = False,
):
    """
    Randomly select states from available reward systems, convert them to feature
    vectors using state_to_vector(), and save them as data/canonical_states_*.npy.
    Only creates the file if it doesn't already exist (unless force_recreate=True).

    Args:
        states_folder: Name of the folder containing states (e.g., "states_16").
        canonical_states: Total number of canonical states to collect (default: 64).
        run_dir: Path to a state_runs/<spec> folder; if provided, overrides states_folder.
        reward_systems: Optional list of reward systems to sample from.
        force_recreate: If True, regenerate even if the file exists.

    Returns:
        numpy array of shape (canonical_states, STATE_HEAD_LEN + GMAX) containing the canonical
        state feature vectors, or None if file already exists
    """
    if run_dir:
        base_dir = run_dir
        run_name = os.path.basename(os.path.normpath(run_dir))
        output_path = os.path.join(
            os.getcwd(), "data", f"canonical_states_{run_name}.npy"
        )
    else:
        base_dir = os.getcwd()
        output_path = os.path.join(
            base_dir, "data", f"canonical_states_{states_folder}.npy"
        )

    # Check if file already exists
    if os.path.exists(output_path) and not force_recreate:
        cached = np.load(output_path)
        if cached.shape[0] != canonical_states:
            print(
                f"Warning: requested {canonical_states} canonical states, using existing "
                f"{cached.shape[0]} from {output_path}."
            )
        else:
            print(
                f"{os.path.basename(output_path)} already exists at {output_path}. Skipping creation."
            )
        return cached
    if os.path.exists(output_path) and force_recreate:
        print(f"Recreating canonical states at {output_path}.")

    if reward_systems is None:
        if run_dir:
            reward_systems = [
                d
                for d in os.listdir(run_dir)
                if os.path.isdir(os.path.join(run_dir, d))
            ]
        else:
            reward_systems = ["gold", "path"]

    # Calculate number of states per policy
    per_policy = canonical_states // len(reward_systems)
    remainder = canonical_states % len(reward_systems)

    all_states = []

    for idx, policy in enumerate(reward_systems):
        target_count = per_policy + (1 if idx < remainder else 0)
        policy_states = []
        if run_dir:
            pattern = os.path.join(run_dir, policy, "**", "episode_states.npy")
        else:
            pattern = os.path.join(
                base_dir, states_folder, policy, "**", "episode_states.npy"
            )
        policy_files = glob.glob(pattern, recursive=True)
        random.shuffle(policy_files)

        for npy_path in policy_files:
            if len(policy_states) >= target_count:
                break
            try:
                states = np.load(npy_path)
                hazard_path = npy_path.replace(
                    "episode_states.npy", "episode_hazard_counts.npy"
                )
                hazard_counts = None
                if os.path.exists(hazard_path):
                    hazard_counts = np.load(hazard_path)
                if states.ndim == 3:
                    if len(states) > 0:
                        random_idx = random.randint(0, len(states) - 1)
                        state = states[random_idx]
                        hazard_steps = (
                            hazard_counts[random_idx]
                            if hazard_counts is not None
                            else None
                        )
                        feature_vec = state_to_vector(
                            state, hazard_steps=hazard_steps
                        )
                        policy_states.append(feature_vec)
                elif states.ndim == 2:
                    hazard_steps = None
                    if hazard_counts is not None and len(hazard_counts) > 0:
                        hazard_steps = hazard_counts[0]
                    feature_vec = state_to_vector(states, hazard_steps=hazard_steps)
                    policy_states.append(feature_vec)
            except Exception as e:
                print(f"Error converting state to vector from {npy_path}: {e}")
                continue

        if len(policy_states) < target_count:
            print(
                f"Warning: Only collected {len(policy_states)} states for {policy} (need {target_count})"
            )
        all_states.extend(policy_states)

    canonical_states = np.array(all_states, dtype=np.float32)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    np.save(output_path, canonical_states)
    print(f"Saved {len(canonical_states)} canonical states to {output_path}")

    return canonical_states


def process_states(
    states_folder: str = "states_16",
    run_dir: str | None = None,
    reward_systems: list[str] | None = None,
    percentages: list[float] | None = None,
):
    """
    Process states from all available reward systems and save to data/processed_states*.csv.
    Transitions are stored in memory as tuples of numpy arrays (np.array, np.array).
    If the file already exists, returns it as a pandas DataFrame without reprocessing.

    Args:
        states_folder: Name of the folder containing states (e.g., "states_16").
        run_dir: Path to a state_runs/<spec> folder; if provided, overrides states_folder.
        reward_systems: Optional list of reward systems to include.
        percentages: Optional list of snapshot percentages to sample.

    Returns:
        pandas DataFrame with columns: policy_target, policy_name, reward, transitions
        where transitions is a list of tuples (np.array, np.array)
    """
    base_dir = os.getcwd()
    if run_dir:
        run_name = os.path.basename(os.path.normpath(run_dir))
        output_path = os.path.join(base_dir, "data", f"processed_states_{run_name}.csv")
        transitions_path = os.path.join(
            base_dir, "data", f"processed_states_transitions_{run_name}.pkl"
        )
    else:
        output_path = os.path.join(base_dir, "data", "processed_states.csv")
        transitions_path = os.path.join(
            base_dir, "data", "processed_states_transitions.pkl"
        )

    # Check if file already exists
    if os.path.exists(output_path) and os.path.exists(transitions_path):
        print(
            f"processed_states.csv already exists at {output_path}. Loading and returning."
        )
        try:
            if os.path.getsize(output_path) == 0:
                raise pd.errors.EmptyDataError("empty processed_states.csv")
            df = pd.read_csv(output_path)
            if df.empty or len(df.columns) == 0:
                raise pd.errors.EmptyDataError("empty processed_states.csv")
            # Load transitions from pickle file
            with open(transitions_path, "rb") as f:
                transitions_list = pickle.load(f)
            df["transitions"] = transitions_list
            return df
        except pd.errors.EmptyDataError:
            print(
                f"Warning: {output_path} is empty. Regenerating processed states."
            )
        except Exception as e:
            print(
                f"Warning: failed to load processed states from {output_path}: {e}. Regenerating."
            )
        # Cleanup before regenerating
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(transitions_path):
                os.remove(transitions_path)
        except OSError as e:
            print(f"Warning: failed to remove stale processed states files: {e}")

    if reward_systems is None:
        if run_dir:
            reward_systems = [
                d
                for d in os.listdir(run_dir)
                if os.path.isdir(os.path.join(run_dir, d))
            ]
        else:
            reward_systems = ["gold", "path"]
    if percentages is None:
        percentages = [0.2, 0.6, 0.8]

    results = []

    # First, find seeds that exist in BOTH policies
    seed_sets = []
    for policy in reward_systems:
        if run_dir:
            seed_pattern = os.path.join(run_dir, policy, "seed_*")
        else:
            seed_pattern = os.path.join(base_dir, states_folder, policy, "seed_*")
        seed_dirs = glob.glob(seed_pattern)
        seed_sets.append({os.path.basename(d) for d in seed_dirs})

    if seed_sets:
        common_seed_names = set.intersection(*seed_sets)
    else:
        common_seed_names = set()
    common_seed_names = sorted(list(common_seed_names))

    print(f"Found {len(common_seed_names)} common seeds across reward systems")

    # Select only 10% of common seeds
    # num_seeds_to_use = max(1, int(len(common_seed_names) * 0.1))
    # selected_seed_names = random.sample(common_seed_names, num_seeds_to_use)
    selected_seed_names = common_seed_names

    print(f"Using {len(selected_seed_names)} common seeds")
    print(f"Selected seeds: {selected_seed_names}")

    spec_prefix = None
    if run_dir:
        run_name = os.path.basename(os.path.normpath(run_dir))
        if run_name:
            spec_prefix = run_name.split("_")[0]

    # Now process both policies using the same selected seeds
    for policy in reward_systems:
        print(f"\nProcessing policy: {policy}")

        for seed_name in selected_seed_names:
            if run_dir:
                seed_dir = os.path.join(run_dir, policy, seed_name)
            else:
                seed_dir = os.path.join(base_dir, states_folder, policy, seed_name)
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
                energy_j = (
                    row["energy_j"] if "energy_j" in df_rewards.columns else None
                )
                time_s = row["time"] if "time" in df_rewards.columns else None

                npy_path = get_episode_path(
                    run_dir if run_dir else base_dir,
                    policy,
                    seed_name,
                    episode_id,
                    None if run_dir else states_folder,
                )

                if not os.path.exists(npy_path):
                    print(f"Warning: {npy_path} not found. Skipping.")
                    continue

                try:
                    states = np.load(npy_path)
                    hazard_path = npy_path.replace(
                        "episode_states.npy", "episode_hazard_counts.npy"
                    )
                    hazard_counts = None
                    if os.path.exists(hazard_path):
                        hazard_counts = np.load(hazard_path)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue

                # Process states
                state_vectors = []
                for idx, state in enumerate(states):
                    try:
                        hazard_steps = (
                            hazard_counts[idx] if hazard_counts is not None else None
                        )
                        vec = state_to_vector(state, hazard_steps=hazard_steps)
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
                # policy_name: {spec_} {target}_{seed_number}_{percent}
                base_name = (
                    f"{policy}_{seed_name.replace('seed_', '')}_{int(p * 100)}"
                )
                policy_name = f"{spec_prefix}_{base_name}" if spec_prefix else base_name

                results.append(
                    {
                        "policy_target": policy,
                        "policy_name": policy_name,
                        "reward": reward,
                        "transitions": pairs,
                        "episode_id": episode_id,
                        "seed_name": seed_name,
                        "energy_j": energy_j,
                        "time_s": time_s,
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
                "energy_j": result.get("energy_j"),
                "time_s": result.get("time_s"),
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
