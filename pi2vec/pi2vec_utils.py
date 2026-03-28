import glob
import json
import os
import pickle
import random
from functools import lru_cache

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


def _build_minigrid_value_map(minigrid_ids: dict) -> dict[int, int]:
    object_to_idx = minigrid_ids.get("object_to_idx", {}) if minigrid_ids else {}
    aliases = minigrid_ids.get("aliases", {}) if minigrid_ids else {}

    def _idx(name: str) -> int | None:
        val = aliases.get(name)
        if val is not None:
            return int(val)
        val = object_to_idx.get(name)
        if val is not None:
            return int(val)
        return None

    mapping = {}
    for name, value in (
        ("wall", BLOCK_POSITION_VALUE),
        ("ball", GOLD_POSITION_VALUE),
        ("gold", GOLD_POSITION_VALUE),
        ("lava", HAZARD_POSITION_VALUE),
        ("hazard", HAZARD_POSITION_VALUE),
        ("key", LEVER_POSITION_VALUE),
        ("lever", LEVER_POSITION_VALUE),
        ("goal", TARGET_POSITION_VALUE),
        ("exit", TARGET_POSITION_VALUE),
    ):
        idx = _idx(name)
        if idx is not None:
            mapping[idx] = value
    return mapping


def _minigrid_state_to_gridworld(
    state: dict,
    value_map: dict[int, int],
) -> np.ndarray:
    grid_data = state.get("grid")
    if grid_data is None:
        raise ValueError("Missing 'grid' in minigrid state.")
    grid = np.array(grid_data)
    if grid.ndim == 3:
        obj_layer = grid[:, :, 0]
    elif grid.ndim == 2:
        obj_layer = grid
    else:
        raise ValueError(f"Unexpected grid shape: {grid.shape}")

    # Grid.encode is indexed as [x, y]; transpose to [y, x] for row/col layout.
    obj_layer = obj_layer.T

    gridworld = np.zeros(obj_layer.shape, dtype=np.int32)
    for obj_idx, value in value_map.items():
        gridworld[obj_layer == obj_idx] = value

    agent_pos = state.get("agent_pos")
    if agent_pos is not None and len(agent_pos) >= 2:
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        if 0 <= ay < gridworld.shape[0] and 0 <= ax < gridworld.shape[1]:
            gridworld[ay, ax] = AGENT_POSITION_VALUE

    return gridworld


def state_to_vector(
    state: np.ndarray | dict,
    hazard_steps: float | None = None,
    minigrid_ids: dict | None = None,
    minigrid_value_map: dict[int, int] | None = None,
) -> np.ndarray:
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
    if minigrid_ids is not None and isinstance(state, dict):
        if minigrid_value_map is None:
            minigrid_value_map = _build_minigrid_value_map(minigrid_ids)
        state = _minigrid_state_to_gridworld(state, minigrid_value_map)

    if not isinstance(state, np.ndarray):
        raise ValueError("state_to_vector expects a numpy array or minigrid state dict.")

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


@lru_cache(maxsize=1)
def _load_vit_components():
    from transformers import ViTImageProcessor, ViTModel

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    model.eval()
    return processor, model


def _render_env_image(env) -> np.ndarray:
    frame = None
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_frame"):
        try:
            frame = env.unwrapped.get_frame(highlight=False)
        except TypeError:
            frame = env.unwrapped.get_frame()
    if frame is None:
        frame = env.render()
    if frame is None:
        raise ValueError("Env render returned None; ensure render_mode='rgb_array'.")
    if frame.shape[0] > frame.shape[1]:
        # Crop any top banner to keep the square grid area.
        frame = frame[-frame.shape[1] :, :, :]
    return frame


def seed_to_image_array(
    seed: int,
    size: int = 16,
    num_balls: int = 25,
    num_walls: int = 15,
    num_lava: int = 15,
    reward_system: str = "path-gold-hazard-lever",
    ball_reward: float = 0.1,
    key_reward: float = 0.1,
    exit_reward: float = 20.0,
    exit_with_key_reward: float = 40.0,
    lava_penalty: float = -1.0,
    step_penalty: float = -0.001,
    path_progress_scale: float = 0.5,
) -> np.ndarray:
    """Create a LeverGrid RGB image array for a specific environment seed."""
    from policy_reusability.env.lever_minigrid import LeverGridEnv

    env = LeverGridEnv(
        size=size,
        num_balls=num_balls,
        num_walls=num_walls,
        num_lava=num_lava,
        reward_system=reward_system,
        ball_reward=ball_reward,
        key_reward=key_reward,
        exit_reward=exit_reward,
        exit_with_key_reward=exit_with_key_reward,
        lava_penalty=lava_penalty,
        step_penalty=step_penalty,
        path_progress_scale=path_progress_scale,
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=int(seed))
        return _render_env_image(env)
    finally:
        env.close()


def state_to_vector_deep(image: np.ndarray, device: str | None = None) -> np.ndarray:
    """Encode an RGB image array into a ViT feature vector (CLS embedding)."""
    from PIL import Image
    import torch

    processor, model = _load_vit_components()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = next(model.parameters()).device
    if model_device.type != device:
        model = model.to(device)

    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.fromarray(image)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if outputs.pooler_output is not None:
        embedding = outputs.pooler_output
    else:
        embedding = outputs.last_hidden_state[:, 0]
    return embedding.squeeze(0).detach().cpu().numpy()


def _load_eval_seeds(seeds_path: str) -> list[int]:
    with open(seeds_path, "r", encoding="utf-8") as handle:
        seeds = json.load(handle)
    return [int(seed) for seed in seeds]


def _select_top_snapshots(rewards_path: str, top_k: int = 5) -> list[dict]:
    df = pd.read_csv(rewards_path)
    df["episode"] = pd.to_numeric(df.get("episode"), errors="coerce")
    df["timesteps"] = pd.to_numeric(df.get("timesteps"), errors="coerce")
    df["reward"] = pd.to_numeric(df.get("reward"), errors="coerce")
    df["reward_det"] = pd.to_numeric(df.get("reward_det"), errors="coerce")
    df = df.dropna(subset=["episode", "reward", "reward_det"])
    if df.empty:
        return []
    df = df.sort_values(
        ["reward", "reward_det", "episode"], ascending=[False, False, False]
    )
    snapshots = []
    for _, row in df.head(top_k).iterrows():
        snapshot_id = row.get("timesteps")
        if pd.isna(snapshot_id):
            snapshot_id = row["episode"]
        snapshots.append(
            {
                "episode": int(row["episode"]),
                "timesteps": None if pd.isna(row.get("timesteps")) else int(row["timesteps"]),
                "snapshot_id": int(snapshot_id),
                "reward": float(row["reward"]),
                "reward_det": float(row["reward_det"]),
            }
        )
    return snapshots


def create_deeprl_transition_dataset(
    deeprl_root: str = "deeprl_runs/16",
    policies: list[str] | None = None,
    eval_seeds_path: str = "deeprl_runs/16/eval_env_seeds.json",
    output_path: str = "data_rl/deeprl_transitions.pkl",
    canonical_states_path: str = "data_rl/deeprl_canonical_states.pkl",
    canonical_count: int = 128,
    minigrid_ids_path: str | None = None,
    size: int = 16,
    num_balls: int = 25,
    num_walls: int = 8,
    num_lava: int = 8,
    ball_reward: float = 0.1,
    key_reward: float = 0.1,
    exit_reward: float = 50.0,
    exit_with_key_reward: float = 100.0,
    lava_penalty: float = -1.0,
    step_penalty: float = -0.001,
    path_progress_scale: float = 1.0,
    device: str | None = None,
    deterministic: bool = True,
    show_progress: bool = True,
    regressor_only_fraction: float = 0.6,
):
    """
    Build a dataset of (s, s') transitions for deep RL policies and save to a pickle file.

    Transitions are loaded from per-snapshot eval files, not re-generated via env rollouts.
    States are encoded via state_to_vector using MiniGrid ID mappings.
    """
    from tqdm import tqdm

    if policies is None:
        policies = [
            "hazard",
            "lever",
            "path",
            "gold",
            "hazard-lever",
            "path-gold",
            "path-gold-hazard",
        ]

    _ = eval_seeds_path  # unused: transitions are read from snapshot files
    _ = (size, num_balls, num_walls, num_lava)
    _ = (
        ball_reward,
        key_reward,
        exit_reward,
        exit_with_key_reward,
        lava_penalty,
        step_penalty,
        path_progress_scale,
        device,
        deterministic,
    )
    label_by_rank = [100, 80, 60, 40, 20]
    dataset = []
    canonical_states = []
    seen_states = 0

    if not (0.0 <= regressor_only_fraction <= 1.0):
        raise ValueError("regressor_only_fraction must be between 0 and 1.")

    def maybe_add_canonical(vec):
        nonlocal seen_states
        seen_states += 1
        if len(canonical_states) < canonical_count:
            canonical_states.append(vec)
        else:
            idx = random.randrange(seen_states)
            if idx < canonical_count:
                canonical_states[idx] = vec

    if not minigrid_ids_path:
        raise ValueError("minigrid_ids_path is required for deep RL state encoding.")
    try:
        with open(minigrid_ids_path, "r", encoding="utf-8") as handle:
            minigrid_ids = json.load(handle)
        minigrid_value_map = _build_minigrid_value_map(minigrid_ids)
    except Exception as e:
        raise ValueError(f"Failed to load minigrid ids from {minigrid_ids_path}: {e}")
    if not minigrid_value_map:
        raise ValueError(
            f"No object mappings found in {minigrid_ids_path}; cannot encode states."
        )

    for policy in policies:
        policy_dir = os.path.join(deeprl_root, policy)
        rewards_path = os.path.join(policy_dir, "episode_rewards.csv")
        if not os.path.exists(rewards_path):
            print(f"Warning: missing {rewards_path}; skipping {policy}.")
            continue

        snapshots = _select_top_snapshots(rewards_path, top_k=5)
        if not snapshots:
            print(f"Warning: no rewards found in {rewards_path}; skipping {policy}.")
            continue

        regressor_only_count = max(
            1, int(round(len(snapshots) * regressor_only_fraction))
        )
        regressor_only_indices = set(
            random.sample(range(len(snapshots)), regressor_only_count)
        )

        for rank, snapshot in enumerate(snapshots):
            label = (
                label_by_rank[rank] if rank < len(label_by_rank) else label_by_rank[-1]
            )
            episode_id = snapshot["snapshot_id"]
            reward = snapshot["reward"]
            reward_det = snapshot.get("reward_det")
            episode_dir = os.path.join(
                policy_dir, "episodes", f"episode_{episode_id:06d}"
            )
            transitions_path = os.path.join(episode_dir, "eval_transitions.pkl")
            model_path = os.path.join(episode_dir, "model.zip")
            if not os.path.exists(transitions_path):
                print(f"Warning: missing {transitions_path}; skipping {policy}.")
                continue
            if not os.path.exists(model_path):
                print(f"Warning: missing {model_path}; skipping {policy}.")
                continue

            with open(transitions_path, "rb") as handle:
                raw_transitions = pickle.load(handle)

            if not raw_transitions:
                print(f"Warning: empty transitions in {transitions_path}; skipping.")
                continue

            policy_label = f"{policy}_{label}"
            print(f"Encoding transitions for {policy_label} (episode {episode_id})")
            transitions = []
            iterator = raw_transitions
            if show_progress:
                iterator = tqdm(
                    raw_transitions,
                    total=len(raw_transitions),
                    desc=policy_label,
                )
            for item in iterator:
                state = item.get("state", {})
                next_state = item.get("next_state", {})
                try:
                    prev_vec = state_to_vector(
                        state,
                        hazard_steps=None,
                        minigrid_ids=minigrid_ids,
                        minigrid_value_map=minigrid_value_map,
                    )
                    next_vec = state_to_vector(
                        next_state,
                        hazard_steps=None,
                        minigrid_ids=minigrid_ids,
                        minigrid_value_map=minigrid_value_map,
                    )
                except Exception as e:
                    print(f"Warning: failed to encode transition: {e}")
                    continue
                transitions.append((prev_vec, next_vec))
                maybe_add_canonical(prev_vec)
                maybe_add_canonical(next_vec)

            dataset.append(
                {
                    "model_name": policy_label,
                    "policy_target": policy,
                    "reward": reward,
                    "reward_det": reward_det,
                    "transitions": transitions,
                    "episode_id": episode_id,
                    "episode_index": snapshot.get("episode"),
                    "timesteps": snapshot.get("timesteps"),
                    "model_path": model_path,
                    "include_in_vdb": rank not in regressor_only_indices,
                }
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved deep RL transitions to {output_path}")
    if canonical_states:
        os.makedirs(os.path.dirname(canonical_states_path), exist_ok=True)
        canonical_array = np.array(canonical_states, dtype=np.float32)
        with open(canonical_states_path, "wb") as handle:
            pickle.dump(canonical_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved deep RL canonical states to {canonical_states_path}")
    else:
        print("Warning: no canonical states sampled; transitions were empty.")
    return dataset


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
    output_suffix: str | None = None,
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
    suffix = f"_{output_suffix}" if output_suffix else ""
    if run_dir:
        base_dir = run_dir
        run_name = os.path.basename(os.path.normpath(run_dir))
        output_path = os.path.join(
            os.getcwd(), "data", f"canonical_states_{run_name}{suffix}.npy"
        )
    else:
        base_dir = os.getcwd()
        output_path = os.path.join(
            base_dir, "data", f"canonical_states_{states_folder}{suffix}.npy"
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
        spec_prefix = run_name.split("_")[0] if run_name else None
        output_path = os.path.join(base_dir, "data", f"processed_states_{run_name}.csv")
        transitions_path = os.path.join(
            base_dir, "data", f"processed_states_transitions_{run_name}.pkl"
        )
    else:
        spec_prefix = None
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
            if spec_prefix and "spec" not in df.columns:
                df["spec"] = spec_prefix
            # Load transitions from pickle file
            with open(transitions_path, "rb") as f:
                transitions_list = pickle.load(f)
            df["transitions"] = transitions_list
            # Filter by reward_systems if specified (prevents experiment leakage)
            if reward_systems is not None:
                df = df[df["policy_target"].isin(reward_systems)].reset_index(drop=True)
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
        percentages = [0.2, 0.6, 1.0]

    # Track X5 extra-snapshot counts so X10 can reuse the same absolute counts.
    # Keyed by (policy, seed_name).
    if not hasattr(process_states, "_x5_extra_counts"):
        process_states._x5_extra_counts = {}
    x5_extra_counts = process_states._x5_extra_counts

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

            df_rewards["episode"] = pd.to_numeric(
                df_rewards.get("episode"), errors="coerce"
            )
            df_rewards["reward"] = pd.to_numeric(
                df_rewards.get("reward"), errors="coerce"
            )
            df_rewards = df_rewards.dropna(subset=["episode", "reward"])
            if df_rewards.empty:
                continue

            episode_glob = os.path.join(
                seed_dir, "episodes", "episode_*", "episode_states.npy"
            )
            available_episode_ids = set()
            for path in glob.glob(episode_glob):
                episode_dir = os.path.basename(os.path.dirname(path))
                if not episode_dir.startswith("episode_"):
                    continue
                try:
                    available_episode_ids.add(int(episode_dir.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if available_episode_ids:
                df_rewards = df_rewards[
                    df_rewards["episode"].isin(available_episode_ids)
                ]
                if df_rewards.empty:
                    print(
                        f"Warning: no episode snapshots found for {seed_dir}. Skipping."
                    )
                    continue

            reward_values = sorted(df_rewards["reward"].unique().tolist())
            min_reward = reward_values[0]
            mid_reward = reward_values[len(reward_values) // 2]
            max_reward = reward_values[-1]

            sample_labels = [int(p * 100) for p in percentages]
            if len(sample_labels) != 3:
                sample_labels = [20, 60, 100]
            label_by_key = {
                "min": sample_labels[0],
                "mid": sample_labels[1],
                "max": sample_labels[2],
            }

            selection_plan = [
                ("max", max_reward, True),  # best snapshot (max reward, latest episode)
                ("min", min_reward, True),
                ("mid", mid_reward, False),
            ]
            used_episodes: set[int] = set()
            selected_rows = {}

            for key, reward_value, prefer_max in selection_plan:
                candidates = df_rewards[df_rewards["reward"] == reward_value].copy()
                if candidates.empty:
                    continue
                candidates = candidates.sort_values(
                    "episode", ascending=not prefer_max
                )
                row = None
                for _, cand in candidates.iterrows():
                    episode_id = int(cand["episode"])
                    if episode_id in used_episodes:
                        continue
                    row = cand
                    used_episodes.add(episode_id)
                    break
                if row is None:
                    remaining = df_rewards[
                        ~df_rewards["episode"].isin(used_episodes)
                    ].copy()
                    if remaining.empty:
                        print(
                            "Warning: insufficient distinct episodes for reward selection."
                        )
                        continue
                    remaining = remaining.sort_values(
                        "episode", ascending=not prefer_max
                    )
                    row = remaining.iloc[0]
                    used_episodes.add(int(row["episode"]))
                selected_rows[key] = row

            # Add 60% random snapshots (excluding episode 0 and already-selected episodes).
            # For X10, reuse the absolute count selected for X5 (if available),
            # instead of taking 60% of the larger pool.
            extra_rows = []
            random_pool = df_rewards[
                (~df_rewards["episode"].isin(used_episodes))
                & (df_rewards["episode"] != 0)
            ].copy()
            if not random_pool.empty:
                if spec_prefix == "X10":
                    sample_count = x5_extra_counts.get((policy, seed_name))
                    if sample_count is None:
                        sample_count = int(len(random_pool) * 0.6)
                else:
                    sample_count = int(len(random_pool) * 0.6)
                    if spec_prefix == "X5":
                        x5_extra_counts[(policy, seed_name)] = sample_count
                sample_count = min(sample_count, len(random_pool))
                if sample_count > 0:
                    seed_int = int(seed_name.replace("seed_", ""))
                    extra_rows = (
                        random_pool.sample(n=sample_count, random_state=seed_int)
                        .sort_values("episode")
                        .to_dict("records")
                    )

            rows_to_process = []
            for key in ("min", "mid", "max"):
                row = selected_rows.get(key)
                if row is None:
                    continue
                label = label_by_key.get(key)
                rows_to_process.append((row, label))
            for row in extra_rows:
                episode_id = int(row["episode"])
                label = f"ep{episode_id}"
                rows_to_process.append((row, label))

            for row, label in rows_to_process:
                if row is None or label is None:
                    continue
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
                base_name = f"{policy}_{seed_name.replace('seed_', '')}_{label}"
                policy_name = f"{spec_prefix}_{base_name}" if spec_prefix else base_name
                include_in_vdb = not str(label).startswith("ep")

                results.append(
                    {
                        "spec": spec_prefix,
                        "policy_target": policy,
                        "policy_name": policy_name,
                        "reward": reward,
                        "transitions": pairs,
                        "episode_id": episode_id,
                        "seed_name": seed_name,
                        "energy_j": energy_j,
                        "time_s": time_s,
                        "include_in_vdb": include_in_vdb,
                    }
                )

    # Extract transitions before creating DataFrame (to keep them as tuples of numpy arrays)
    transitions_list = [result["transitions"] for result in results]

    # Create DataFrame without transitions column (for CSV storage)
    df_metadata = pd.DataFrame(
        [
            {
                "policy_target": result["policy_target"],
                "spec": result.get("spec"),
                "policy_name": result["policy_name"],
                "reward": result["reward"],
                "episode_id": result["episode_id"],
                "seed_name": result["seed_name"],
                "energy_j": result.get("energy_j"),
                "time_s": result.get("time_s"),
                "include_in_vdb": result.get("include_in_vdb", True),
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
