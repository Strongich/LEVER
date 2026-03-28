"""
Full experiment runner for PPO policy composition using GPI.

GPI (Generalised Policy Improvement) for PPO:
  V_best = max(V1(s), V2(s), ...)
  a* = actor_best(s)   (actor from the model with highest V(s))

Output CSV columns:
  setup, spec, seed, query, scratch_reward, scratch_time_s,
  targeted_reward, exhaustive_reward, hybrid_reward,
  targeted_policies, exhaustive_policies, hybrid_policies,
  decomp_time, targeted_time, exhaustive_time, hybrid_time

Scoring approach (exhaustive / hybrid):
  1. Pre-compute initial env states for eval seeds (one-time env creation).
  2. For each candidate combo, simulate GPI rollouts on cached states
     (no env creation), collect encoded transitions, train SF model,
     compute embedding, predict reward via regressor.
  3. Pick best combo by regressor score.
  4. Evaluate the winner with actual env interaction for reported reward.

Scratch baseline:
  - scratch_reward: reward_det from best snapshot according to --prefilter.
    v1: best positive snapshot (reward>0 AND reward_det>0, sorted descending).
    v2: same ranking, but no positivity filter.
  - scratch_time: last CSV record's time_seconds.
"""

import argparse
import itertools
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO

from config import (
    DOUBLE_POLICIES,
    GRIDWORLD_AVAILABLE_ACTIONS,
    TRIPLE_POLICIES,
    TRIVIAL_POLICIES,
)
from tabular.full_experiment import decompose_query_with_retry, score_candidates
from pi2vec.pi2vec_utils import _build_minigrid_value_map, state_to_vector
from pi2vec.psimodel import SuccessorFeatureModelDeep
from pi2vec.train_successor import train_and_save_successor_model_variants
from policy_reusability.data_generation.deeprl.train_ppo import (
    MinigridFeaturesExtractor,  # noqa: F401 – needed so PPO.load can unpickle the CNN
    _extract_full_state,
    _maybe_transpose_obs,
    _wrap_obs,
)
from policy_reusability.env.lever_minigrid import LeverGridEnv
from search_faiss_policies import PolicyRetriever

# ── Grid-8 preset (matches train_ppo grid8) ─────────────────────────────────
GRID_SIZE = 8
NUM_BALLS = 6
NUM_WALLS = 4
NUM_LAVA = 4
OBS_MODE = "local"
LOCAL_SIZE = 7
MAX_STEPS = 2 * GRID_SIZE * GRID_SIZE  # 128


def set_grid_params(grid_size: int):
    """Update module-level grid size and derived constants."""
    global GRID_SIZE, MAX_STEPS
    GRID_SIZE = grid_size
    MAX_STEPS = 2 * grid_size * grid_size


# ── Observation params ───────────────────────────────────────────────────────
def set_obs_params(obs_mode: str, local_size: int):
    """Update module-level observation settings."""
    global OBS_MODE, LOCAL_SIZE
    OBS_MODE = obs_mode
    LOCAL_SIZE = local_size

# MiniGrid object-type IDs (from OBJECT_TO_IDX)
OBJ_EMPTY = 1
OBJ_WALL = 2
OBJ_KEY = 5
OBJ_BALL = 6
OBJ_GOAL = 8
OBJ_LAVA = 9

# Reward defaults (matching train_ppo.py)
BALL_REWARD = 15.0
KEY_REWARD = 20.0
EXIT_REWARD = 50.0
EXIT_WITH_KEY_REWARD = 100.0
LAVA_PENALTY = -1.0
STEP_PENALTY = -0.05
PATH_PROGRESS_SCALE = 1.0

# ── Experiment definitions (same queries as tabular/full_experiment.py) ──────
TRIVIAL_EXPERIMENTS = [
    {
        "setup": "path-gold",
        "query": "Find the fastest exit and collect as much gold as possible",
        "expected_count": 2,
    },
    {
        "setup": "path-gold-hazard",
        "query": "Find the fastest exit, collect as much gold as possible, avoid hazards at all cost",
        "expected_count": 3,
    },
    {
        "setup": "path-gold-hazard-lever",
        "query": "Find the fastest exit after activating the lever, collect as much gold as possible, avoid hazards at all cost",
        "expected_count": 4,
    },
]

DOUBLE_EXPERIMENTS = [
    {
        "setup": "path-gold-hazard",
        "query": "Find the fastest exit, collect as much gold as possible, avoid hazards at all cost",
        "expected_count": 2,
    },
    {
        "setup": "path-gold-hazard-lever",
        "query": "Find the fastest exit after activating the lever, collect as much gold as possible, avoid hazards at all cost",
        "expected_count": 2,
    },
]

TRIPLE_EXPERIMENTS = [
    {
        "setup": "path-gold-hazard-lever",
        "query": "Find the fastest exit after activating the lever, collect as much gold as possible, avoid hazards at all cost",
        "expected_count": 2,
    },
]

# ── Model cache ──────────────────────────────────────────────────────────────
_model_cache: dict[str, PPO] = {}


def get_ppo_model(model_path: str, device: str = "cpu") -> PPO:
    """Load a PPO model, caching to avoid redundant disk reads."""
    if model_path not in _model_cache:
        _model_cache[model_path] = PPO.load(model_path, device=device)
    return _model_cache[model_path]


# ── Observation from cached grid state ───────────────────────────────────────
def obs_from_state(
    grid: np.ndarray, agent_pos: tuple, local_size: int | None = None
) -> np.ndarray:
    """Replicate LocalObsWrapper + ScaleObsWrapper without an env instance.

    Returns a (3, local_size, local_size) uint8 array ready for the PPO CNN.
    """
    if local_size is None:
        local_size = LOCAL_SIZE
    half = local_size // 2
    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    W, H = grid.shape[0], grid.shape[1]
    wall = np.array([OBJ_WALL, 0, 0], dtype=np.uint8)

    patch = np.zeros((local_size, local_size, 3), dtype=np.uint8)
    for dx in range(local_size):
        for dy in range(local_size):
            gx = ax - half + dx
            gy = ay - half + dy
            if 0 <= gx < W and 0 <= gy < H:
                patch[dx, dy] = grid[gx, gy]
            else:
                patch[dx, dy] = wall

    # Channel-first (3, H, W) — matches LocalObsWrapper output
    patch = patch.transpose(2, 0, 1)
    # ScaleObsWrapper: [25, 51, 127] per channel
    scales = np.array([25, 51, 127], dtype=np.uint8)[:, None, None]
    return (patch.astype(np.uint16) * scales).clip(0, 255).astype(np.uint8)


# ── Simulated cardinal step ──────────────────────────────────────────────────
# MiniGrid cardinal dirs: 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
_DELTAS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def simulate_step(
    grid: np.ndarray, agent_pos: tuple, action: int
) -> tuple[np.ndarray, tuple, bool]:
    """Simulate one cardinal action on the encoded grid.

    Returns (new_grid, new_agent_pos, terminated).
    Objects (ball/key) are removed when the agent steps onto them.
    """
    dx, dy = _DELTAS[action]
    nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
    W, H = grid.shape[0], grid.shape[1]

    if nx < 0 or ny < 0 or nx >= W or ny >= H:
        return grid, agent_pos, False

    cell_type = int(grid[nx, ny, 0])

    if cell_type == OBJ_WALL:
        return grid, agent_pos, False

    new_grid = grid.copy()
    terminated = False

    if cell_type == OBJ_BALL:
        new_grid[nx, ny] = [OBJ_EMPTY, 0, 0]
    elif cell_type == OBJ_KEY:
        new_grid[nx, ny] = [OBJ_EMPTY, 0, 0]
    elif cell_type == OBJ_LAVA:
        terminated = True
    elif cell_type == OBJ_GOAL:
        terminated = True

    return new_grid, (nx, ny), terminated


# ── GPI action selection (critic-based actor selection for PPO) ───────────────
def gpi_action_ppo(models: list[PPO], obs: np.ndarray, device: str = "cpu") -> int:
    """GPI for PPO: evaluate V(s) per model's critic, pick actor with highest V."""
    obs_tensor = torch.as_tensor(obs[np.newaxis]).float().to(device)
    best_value = -float("inf")
    best_model = models[0]
    for model in models:
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor)  # (1, 1)
            v = value.item()
        if v > best_value:
            best_value = v
            best_model = model
    action, _ = best_model.predict(obs, deterministic=True)
    return int(action)


# ── Simulated GPI rollout (no env creation) ──────────────────────────────────
def gpi_rollout_simulated(
    models: list[PPO],
    initial_state: dict,
    minigrid_ids: dict,
    minigrid_value_map: dict,
    max_steps: int | None = None,
    local_size: int | None = None,
    device: str = "cpu",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run GPI on a cached grid state, collecting encoded (s, s') transitions."""
    if max_steps is None:
        max_steps = MAX_STEPS
    if local_size is None:
        local_size = LOCAL_SIZE
    grid = initial_state["grid"].copy()
    agent_pos = tuple(int(x) for x in initial_state["agent_pos"])
    agent_dir = int(initial_state["agent_dir"])

    transitions: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(max_steps):
        obs = obs_from_state(grid, agent_pos, local_size)
        action = gpi_action_ppo(models, obs, device)

        state_dict = {"grid": grid, "agent_pos": agent_pos, "agent_dir": agent_dir}
        s_vec = state_to_vector(
            state_dict, minigrid_ids=minigrid_ids, minigrid_value_map=minigrid_value_map
        )

        new_grid, new_agent_pos, terminated = simulate_step(grid, agent_pos, action)
        new_agent_dir = action

        next_dict = {
            "grid": new_grid,
            "agent_pos": new_agent_pos,
            "agent_dir": new_agent_dir,
        }
        s_next_vec = state_to_vector(
            next_dict,
            minigrid_ids=minigrid_ids,
            minigrid_value_map=minigrid_value_map,
        )

        transitions.append((s_vec, s_next_vec))

        if terminated:
            break

        grid = new_grid
        agent_pos = new_agent_pos
        agent_dir = new_agent_dir

    return transitions


# ── Pre-compute initial env states for all eval seeds ────────────────────────
def precompute_env_states(
    eval_seeds: list[int], reward_system: str
) -> dict[int, dict]:
    """Create one env per seed, extract its initial state, tear down the env."""
    cache: dict[int, dict] = {}
    for seed in eval_seeds:
        env = LeverGridEnv(
            size=GRID_SIZE,
            num_balls=NUM_BALLS,
            num_walls=NUM_WALLS,
            num_lava=NUM_LAVA,
            reward_system=reward_system,
            ball_reward=BALL_REWARD,
            key_reward=KEY_REWARD,
            exit_reward=200.0 if reward_system == "path" else EXIT_REWARD,
            exit_with_key_reward=EXIT_WITH_KEY_REWARD,
            lava_penalty=LAVA_PENALTY,
            step_penalty=STEP_PENALTY,
            path_progress_scale=PATH_PROGRESS_SCALE,
            action_mode="cardinal",
        )
        env.reset(seed=seed)
        cache[seed] = _extract_full_state(env)
        env.close()
    return cache


# ── Create a wrapped eval env for one seed ───────────────────────────────────
def _make_eval_env(reward_system: str):
    env = LeverGridEnv(
        size=GRID_SIZE,
        num_balls=NUM_BALLS,
        num_walls=NUM_WALLS,
        num_lava=NUM_LAVA,
        reward_system=reward_system,
        ball_reward=BALL_REWARD,
        key_reward=KEY_REWARD,
        exit_reward=200.0 if reward_system == "path" else EXIT_REWARD,
        exit_with_key_reward=EXIT_WITH_KEY_REWARD,
        lava_penalty=LAVA_PENALTY,
        step_penalty=STEP_PENALTY,
        path_progress_scale=PATH_PROGRESS_SCALE,
        action_mode="cardinal",
        render_mode="rgb_array",
    )
    return _wrap_obs(env, OBS_MODE, local_size=LOCAL_SIZE)


# ── Evaluate a single PPO model on one seed ──────────────────────────────────
def evaluate_single_model(
    model: PPO, reward_system: str, seed: int, device: str = "cpu"
) -> float:
    env = _make_eval_env(reward_system)
    obs, _ = env.reset(seed=seed)
    obs = _maybe_transpose_obs(obs)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        obs = _maybe_transpose_obs(obs)
    env.close()
    return total_reward


# ── Evaluate GPI composition on one seed ─────────────────────────────────────
def evaluate_gpi_seed(
    models: list[PPO], reward_system: str, seed: int, device: str = "cpu"
) -> float:
    env = _make_eval_env(reward_system)
    obs, _ = env.reset(seed=seed)
    obs = _maybe_transpose_obs(obs)
    done = False
    total_reward = 0.0
    while not done:
        action = gpi_action_ppo(models, obs, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        obs = _maybe_transpose_obs(obs)
    env.close()
    return total_reward


# ── Embedding from simulated GPI rollout ─────────────────────────────────────
def embedding_from_gpi_rollout(
    models: list[PPO],
    state_cache: dict[int, dict],
    scoring_seeds: list[int],
    canonical_states: np.ndarray,
    setup_name: str,
    minigrid_ids: dict,
    minigrid_value_map: dict,
    device: str = "cpu",
) -> np.ndarray | None:
    """Collect transitions via simulated GPI, train SF, return embedding."""
    all_transitions: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in scoring_seeds:
        if seed not in state_cache:
            continue
        transitions = gpi_rollout_simulated(
            models,
            state_cache[seed],
            minigrid_ids,
            minigrid_value_map,
            device=device,
        )
        all_transitions.extend(transitions)

    if not all_transitions:
        return None

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        _, embeddings = train_and_save_successor_model_variants(
            "gpi_combo_tmp",
            all_transitions,
            {setup_name: canonical_states},
            epochs=50,
            show_progress=False,
            policy_seed="gpi",
            model_cls=SuccessorFeatureModelDeep,
        )
    return embeddings.get(setup_name)


# ── Scratch info from episode_rewards.csv ────────────────────────────────────
def get_scratch_info(
    base_dir: str, reward_system: str, prefilter: str = "v1"
) -> tuple[str | None, float | None, float | None]:
    """Return (best_model_path, scratch_time_s, best_reward_det) for a prefilter."""
    csv_path = os.path.join(base_dir, reward_system, "episode_rewards.csv")
    if not os.path.exists(csv_path):
        return None, None, None

    df = pd.read_csv(csv_path)
    for col in ("reward", "reward_det", "timesteps", "time_seconds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best snapshot according to prefilter (v2 = v1 ranking, no positivity filter)
    if prefilter == "v1":
        candidates = df[(df["reward"] > 0) & (df["reward_det"] > 0)].copy()
    elif prefilter == "v2":
        candidates = df.copy()
    else:
        raise ValueError(f"Unknown prefilter: {prefilter}")

    best_model_path = None
    best_reward_det = None
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["reward", "reward_det"], ascending=[False, False]
        )
        for _, best_row in candidates.iterrows():
            timesteps = int(best_row["timesteps"])
            candidate_path = os.path.join(
                base_dir,
                reward_system,
                "episodes",
                f"episode_{timesteps:06d}",
                "model.zip",
            )
            if os.path.exists(candidate_path):
                best_model_path = candidate_path
                best_reward_det = float(best_row["reward_det"])
                break

    # Scratch time: LAST record's time_seconds
    scratch_time = None
    if (
        "time_seconds" in df.columns
        and not df.empty
        and not df["time_seconds"].isna().all()
    ):
        scratch_time = float(df["time_seconds"].dropna().iloc[-1])

    return best_model_path, scratch_time, best_reward_det


# ── Load candidate model from VDB metadata ───────────────────────────────────
def _load_candidate_model(
    candidate: dict, device: str = "cpu"
) -> PPO | None:
    model_path = candidate.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        return get_ppo_model(model_path, device)
    except Exception as e:
        print(f"  Warning: failed to load {model_path}: {e}")
        return None


# ── Targeted composition ─────────────────────────────────────────────────────
def targeted_composition(
    retriever: PolicyRetriever,
    sub_queries: list[str],
    device: str = "cpu",
) -> tuple[list[PPO] | None, list[str] | None, float]:
    """Best candidate per sub-query (sim > 0.7 + regressor rank), load models."""
    start = time.time()
    models: list[PPO] = []
    policy_names: list[str] = []

    for sq in sub_queries:
        result_dict, _ = retriever.vdb.search_similar_policies(
            sq, policy_seed="ppo"
        )
        results = result_dict.get("results", [])
        results = [r for r in results if r.get("score", 0) > 0.7]
        results = score_candidates(results, retriever)
        results = sorted(
            results, key=lambda x: x.get("regressor_score", -1), reverse=True
        )

        chosen_model = None
        chosen_name = None
        for cand in results:
            m = _load_candidate_model(cand, device)
            if m is not None:
                chosen_model = m
                chosen_name = cand.get("policy_name", "unknown")
                break

        if chosen_model is None:
            print(f"  targeted: no valid model for sub-query '{sq}'")
            return None, None, time.time() - start

        models.append(chosen_model)
        policy_names.append(chosen_name)

    return models, policy_names, time.time() - start


# ── Exhaustive composition ───────────────────────────────────────────────────
def exhaustive_composition(
    retriever: PolicyRetriever,
    sub_queries: list[str],
    state_cache: dict[int, dict],
    scoring_seeds: list[int],
    canonical_states: np.ndarray,
    setup_name: str,
    minigrid_ids: dict,
    minigrid_value_map: dict,
    similarity_threshold: float = 0.7,
    device: str = "cpu",
) -> tuple[list[PPO] | None, list[str] | None, float, float | None]:
    """Try all combos, score via SF embedding + regressor, return best."""
    start = time.time()

    grouped: list[list[dict]] = []
    for sq in sub_queries:
        result_dict, _ = retriever.vdb.search_similar_policies(
            sq, policy_seed="ppo"
        )
        results = result_dict.get("results", [])
        results = [r for r in results if r.get("score", 0) > similarity_threshold]
        grouped.append(results)

    if any(len(g) == 0 for g in grouped):
        print("  exhaustive: insufficient candidates")
        return None, None, time.time() - start, None

    combos = list(itertools.product(*grouped))
    print(f"  exhaustive: {len(combos)} combos")

    best_models: list[PPO] | None = None
    best_names: list[str] | None = None
    best_pred = -float("inf")

    for combo in combos:
        combo_models: list[PPO] = []
        combo_names: list[str] = []
        for cand in combo:
            m = _load_candidate_model(cand, device)
            if m is None:
                break
            combo_models.append(m)
            combo_names.append(cand.get("policy_name", "unknown"))

        if len(combo_models) != len(combo):
            continue

        embedding = embedding_from_gpi_rollout(
            combo_models,
            state_cache,
            scoring_seeds,
            canonical_states,
            setup_name,
            minigrid_ids,
            minigrid_value_map,
            device,
        )
        if embedding is None:
            continue

        if retriever.regressor_model is not None:
            pred = float(
                retriever.regressor_model.predict(
                    np.array(embedding).reshape(1, -1)
                )[0]
            )
        else:
            pred = 0.0

        if pred > best_pred:
            best_pred = pred
            best_models = combo_models
            best_names = combo_names

    elapsed = time.time() - start
    best_pred_out = best_pred if best_pred > -float("inf") else None
    return best_models, best_names, elapsed, best_pred_out


# ── Hybrid composition ───────────────────────────────────────────────────────
def hybrid_composition(
    retriever: PolicyRetriever,
    sub_queries: list[str],
    state_cache: dict[int, dict],
    scoring_seeds: list[int],
    canonical_states: np.ndarray,
    setup_name: str,
    minigrid_ids: dict,
    minigrid_value_map: dict,
    top_k: int = 3,
    similarity_threshold: float = 0.7,
    device: str = "cpu",
) -> tuple[list[PPO] | None, list[str] | None, float, float | None]:
    """Top-k per sub-query by regressor, then exhaustive scoring."""
    start = time.time()

    grouped: list[list[dict]] = []
    for sq in sub_queries:
        result_dict, _ = retriever.vdb.search_similar_policies(
            sq, policy_seed="ppo"
        )
        results = result_dict.get("results", [])
        results = [r for r in results if r.get("score", 0) > similarity_threshold]
        results = score_candidates(results, retriever)
        results = sorted(
            results, key=lambda x: x.get("regressor_score", -1), reverse=True
        )
        grouped.append(results[:top_k])

    if any(len(g) == 0 for g in grouped):
        print("  hybrid: insufficient candidates")
        return None, None, time.time() - start, None

    combos = list(itertools.product(*grouped))
    print(f"  hybrid: top_k={top_k}, {len(combos)} combos")

    best_models: list[PPO] | None = None
    best_names: list[str] | None = None
    best_pred = -float("inf")

    for combo in combos:
        combo_models: list[PPO] = []
        combo_names: list[str] = []
        for cand in combo:
            m = _load_candidate_model(cand, device)
            if m is None:
                break
            combo_models.append(m)
            combo_names.append(cand.get("policy_name", "unknown"))

        if len(combo_models) != len(combo):
            continue

        embedding = embedding_from_gpi_rollout(
            combo_models,
            state_cache,
            scoring_seeds,
            canonical_states,
            setup_name,
            minigrid_ids,
            minigrid_value_map,
            device,
        )
        if embedding is None:
            continue

        if retriever.regressor_model is not None:
            pred = float(
                retriever.regressor_model.predict(
                    np.array(embedding).reshape(1, -1)
                )[0]
            )
        else:
            pred = 0.0

        if pred > best_pred:
            best_pred = pred
            best_models = combo_models
            best_names = combo_names

    elapsed = time.time() - start
    best_pred_out = best_pred if best_pred > -float("inf") else None
    return best_models, best_names, elapsed, best_pred_out


# ── Main experiment runner ───────────────────────────────────────────────────
def run_ppo_experiment(
    output_csv: str,
    base_dir: str = "deeprl_runs/8",
    minigrid_ids_path: str = "deeprl_runs/minigrid_ids.json",
    eval_seeds_path: str = "deeprl_runs/8/eval_env_seeds.json",
    experiments: list[dict] | None = None,
    policy_list: list[str] | None = None,
    index_path: str = "faiss_index_ppo/base/policy_ppo.index",
    metadata_path: str = "faiss_index_ppo/base/metadata_ppo.pkl",
    regressor_model_path: str = "models_ppo/reward_regressor_ppo_base.pkl",
    regressor_variant: str = "base",
    canonical_states_path: str = "data_rl/canonical_states_ppo_base.npy",
    setup_name: str = "base",
    hybrid_top_k: int = 3,
    scoring_n_seeds: int = 10,
    device: str = "cpu",
    prefilter: str = "v1",
):
    """Run targeted / exhaustive / hybrid PPO composition experiments."""

    # Load eval seeds
    with open(eval_seeds_path, "r") as f:
        eval_seeds: list[int] = json.load(f)
    print(f"Loaded {len(eval_seeds)} eval seeds")

    # Seeds used for scoring (simulated rollout); subset for speed
    scoring_seeds = eval_seeds[:scoring_n_seeds]
    print(f"Using {len(scoring_seeds)} seeds for scoring rollouts")

    # Load minigrid IDs
    with open(minigrid_ids_path, "r") as f:
        minigrid_ids: dict = json.load(f)
    minigrid_value_map = _build_minigrid_value_map(minigrid_ids)

    # Load canonical states
    canonical_states = np.load(canonical_states_path)
    print(f"Canonical states: {canonical_states.shape}")

    # Init retriever
    retriever = PolicyRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        regressor_model_path=regressor_model_path,
        regressor_variant=regressor_variant,
        application_name="Grid World",
        available_actions=GRIDWORLD_AVAILABLE_ACTIONS,
    )

    if experiments is None:
        experiments = TRIVIAL_EXPERIMENTS

    rows: list[dict] = []

    for exp in experiments:
        setup = exp["setup"]
        query = exp["query"]
        expected_count = exp.get("expected_count")

        print(f"\n{'=' * 80}")
        print(f"Experiment: {setup} (expected sub-queries: {expected_count})")
        print(f"{'=' * 80}")

        # ── Decompose query ──────────────────────────────────────────────
        sub_queries, decomp_time = decompose_query_with_retry(
            retriever,
            query,
            max_attempts=3,
            expected_count=expected_count,
            policy_list=policy_list,
        )
        print(f"Sub-queries: {sub_queries}")

        # ── Pre-compute env states for scoring ───────────────────────────
        print("Pre-computing env states for scoring...")
        state_cache = precompute_env_states(scoring_seeds, setup)

        # ── Scratch baseline ─────────────────────────────────────────────
        scratch_model_path, scratch_time, scratch_reward_csv = get_scratch_info(
            base_dir, setup, prefilter=prefilter
        )
        scratch_model: PPO | None = None
        if scratch_model_path and os.path.exists(scratch_model_path):
            scratch_model = get_ppo_model(scratch_model_path, device)
            print(f"Scratch model: {scratch_model_path}")
            print(f"Scratch reward_det (CSV): {scratch_reward_csv}")
        else:
            print(f"No scratch model found for {setup}")
        print(f"Scratch time: {scratch_time}")

        # ── Targeted ─────────────────────────────────────────────────────
        print("\nRunning targeted composition...")
        tgt_models, tgt_names, t_targeted = targeted_composition(
            retriever, sub_queries, device
        )
        if tgt_names:
            print(f"  targeted policies: {tgt_names} ({t_targeted:.2f}s)")

        # ── Exhaustive ───────────────────────────────────────────────────
        print("\nRunning exhaustive composition...")
        exh_models, exh_names, t_exhaustive, exh_pred = exhaustive_composition(
            retriever,
            sub_queries,
            state_cache,
            scoring_seeds,
            canonical_states,
            setup_name,
            minigrid_ids,
            minigrid_value_map,
            device=device,
        )
        if exh_names:
            pred_str = f", predicted={exh_pred:.2f}" if exh_pred is not None else ""
            print(f"  exhaustive policies: {exh_names} ({t_exhaustive:.2f}s{pred_str})")

        # ── Hybrid ───────────────────────────────────────────────────────
        print("\nRunning hybrid composition...")
        hyb_models, hyb_names, t_hybrid, hyb_pred = hybrid_composition(
            retriever,
            sub_queries,
            state_cache,
            scoring_seeds,
            canonical_states,
            setup_name,
            minigrid_ids,
            minigrid_value_map,
            top_k=hybrid_top_k,
            device=device,
        )
        if hyb_names:
            pred_str = f", predicted={hyb_pred:.2f}" if hyb_pred is not None else ""
            print(f"  hybrid policies: {hyb_names} ({t_hybrid:.2f}s{pred_str})")

        # ── Evaluate per seed ────────────────────────────────────────────
        print(f"\nEvaluating on {len(eval_seeds)} seeds...")
        for i, seed in enumerate(eval_seeds):
            scratch_reward = None
            targeted_reward = None
            exhaustive_reward = None
            hybrid_reward = None

            if scratch_model is not None:
                scratch_reward = evaluate_single_model(
                    scratch_model, setup, seed, device
                )
            if tgt_models is not None:
                targeted_reward = evaluate_gpi_seed(tgt_models, setup, seed, device)
            if exh_models is not None:
                exhaustive_reward = evaluate_gpi_seed(exh_models, setup, seed, device)
            if hyb_models is not None:
                hybrid_reward = evaluate_gpi_seed(hyb_models, setup, seed, device)

            rows.append(
                {
                    "setup": setup,
                    "spec": "ppo8",
                    "seed": seed,
                    "query": query,
                    "scratch_reward": scratch_reward,
                    "scratch_time_s": scratch_time,
                    "targeted_reward": targeted_reward,
                    "exhaustive_reward": exhaustive_reward,
                    "hybrid_reward": hybrid_reward,
                    "targeted_policies": json.dumps(tgt_names)
                    if tgt_names
                    else None,
                    "exhaustive_policies": json.dumps(exh_names)
                    if exh_names
                    else None,
                    "hybrid_policies": json.dumps(hyb_names)
                    if hyb_names
                    else None,
                    "exhaustive_predicted": exh_pred,
                    "hybrid_predicted": hyb_pred,
                    "decomp_time": round(decomp_time, 6),
                    "targeted_time": round(t_targeted, 6),
                    "exhaustive_time": round(t_exhaustive, 6),
                    "hybrid_time": round(t_hybrid, 6),
                }
            )

            if (i + 1) % 20 == 0:
                print(f"  evaluated {i + 1}/{len(eval_seeds)} seeds")

        # Print aggregate results for this experiment
        seed_rows = [r for r in rows if r["setup"] == setup]

        def _mean(key):
            vals = [r[key] for r in seed_rows if r[key] is not None]
            return sum(vals) / len(vals) if vals else None

        print(f"\n--- {setup} aggregate ({len(seed_rows)} seeds) ---")
        for label, key in [
            ("scratch", "scratch_reward"),
            ("targeted", "targeted_reward"),
            ("exhaustive", "exhaustive_reward"),
            ("hybrid", "hybrid_reward"),
        ]:
            m = _mean(key)
            if m is not None:
                print(f"  {label:12s}: {m:.2f}")
            else:
                print(f"  {label:12s}: N/A")

    # Save CSV
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(rows)} rows to {output_csv}")


# ── Run all three modes ──────────────────────────────────────────────────────
_MODE_VARIANTS = {
    "trivial": {
        "experiments": TRIVIAL_EXPERIMENTS,
        "policy_list": TRIVIAL_POLICIES,
        "variant": "base",
        "setup_name": "base",
    },
    "double": {
        "experiments": DOUBLE_EXPERIMENTS,
        "policy_list": DOUBLE_POLICIES,
        "variant": "pair",
        "setup_name": "pair",
    },
    "triple": {
        "experiments": TRIPLE_EXPERIMENTS,
        "policy_list": TRIPLE_POLICIES,
        "variant": "trip",
        "setup_name": "trip",
    },
}


def _build_mode_config(
    mode: str,
    faiss_base_dir: str = "faiss_index_ppo",
    models_dir: str = "models_ppo",
    data_dir: str = "data_rl",
) -> dict:
    """Construct per-mode config with dynamic base directories."""
    v = _MODE_VARIANTS[mode]
    variant = v["variant"]
    return {
        "experiments": v["experiments"],
        "policy_list": v["policy_list"],
        "index_path": f"{faiss_base_dir}/{variant}/policy_ppo.index",
        "metadata_path": f"{faiss_base_dir}/{variant}/metadata_ppo.pkl",
        "regressor_model_path": f"{models_dir}/reward_regressor_ppo_{variant}.pkl",
        "regressor_variant": variant,
        "canonical_states_path": f"{data_dir}/canonical_states_ppo_{variant}.npy",
        "setup_name": v["setup_name"],
    }


def run_all_modes(
    base_dir: str = "deeprl_runs/8",
    minigrid_ids_path: str = "deeprl_runs/minigrid_ids.json",
    eval_seeds_path: str = "deeprl_runs/8/eval_env_seeds.json",
    results_dir: str = "results_ppo",
    faiss_base_dir: str = "faiss_index_ppo",
    models_dir: str = "models_ppo",
    data_dir: str = "data_rl",
    hybrid_top_k: int = 3,
    scoring_n_seeds: int = 10,
    device: str = "cpu",
    prefilter: str = "v1",
):
    os.makedirs(results_dir, exist_ok=True)

    for mode in _MODE_VARIANTS:
        cfg = _build_mode_config(mode, faiss_base_dir, models_dir, data_dir)
        output_csv = os.path.join(results_dir, f"full_experiment_ppo_{mode}.csv")
        print(f"\n{'#' * 80}")
        print(f"# Mode: {mode}")
        print(f"{'#' * 80}")

        run_ppo_experiment(
            output_csv=output_csv,
            base_dir=base_dir,
            minigrid_ids_path=minigrid_ids_path,
            eval_seeds_path=eval_seeds_path,
            experiments=cfg["experiments"],
            policy_list=cfg["policy_list"],
            index_path=cfg["index_path"],
            metadata_path=cfg["metadata_path"],
            regressor_model_path=cfg["regressor_model_path"],
            regressor_variant=cfg["regressor_variant"],
            canonical_states_path=cfg["canonical_states_path"],
            setup_name=cfg["setup_name"],
            hybrid_top_k=hybrid_top_k,
            scoring_n_seeds=scoring_n_seeds,
            device=device,
            prefilter=prefilter,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run PPO policy composition experiments with GPI."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="deeprl_runs/8",
        help="Directory containing per-reward-system PPO policy folders",
    )
    parser.add_argument(
        "--minigrid-ids-path",
        type=str,
        default="deeprl_runs/minigrid_ids.json",
    )
    parser.add_argument(
        "--eval-seeds-path",
        type=str,
        default="deeprl_runs/8/eval_env_seeds.json",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_ppo",
        help="Base directory for CSV outputs when --mode=all",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Single output CSV (used with --mode != all)",
    )
    parser.add_argument(
        "--mode",
        choices=["trivial", "double", "triple", "all"],
        default="all",
        help="Which experiment variant to run (default: all)",
    )
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    parser.add_argument(
        "--scoring-n-seeds",
        type=int,
        default=10,
        help="Number of eval seeds used for simulated scoring rollouts",
    )
    parser.add_argument(
        "--prefilter",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Scratch prefilter: v1=positive reward filter, v2=no positive filter.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Grid size (updates MAX_STEPS and env params accordingly)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=["partial", "local", "full"],
        default=OBS_MODE,
        help="Observation mode used for evaluation and simulated rollouts",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=LOCAL_SIZE,
        help="Patch side length for --obs-mode local (must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Base directories for FAISS/models/data (used with --mode=all)
    parser.add_argument(
        "--faiss-base-dir",
        type=str,
        default="faiss_index_ppo",
        help="Base dir for FAISS indices (contains base/, pair/, trip/ subdirs)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models_ppo",
        help="Dir for regressor model .pkl files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_rl",
        help="Dir for canonical_states .npy files",
    )

    # Index/regressor overrides (used with single --mode)
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--regressor-model-path", type=str, default=None)
    parser.add_argument("--regressor-variant", type=str, default=None)
    parser.add_argument("--canonical-states-path", type=str, default=None)

    args = parser.parse_args()

    # Update grid size if non-default
    if args.grid_size != GRID_SIZE:
        set_grid_params(args.grid_size)
    if args.obs_mode != OBS_MODE or args.local_size != LOCAL_SIZE:
        set_obs_params(args.obs_mode, args.local_size)

    if args.mode == "all":
        run_all_modes(
            base_dir=args.base_dir,
            minigrid_ids_path=args.minigrid_ids_path,
            eval_seeds_path=args.eval_seeds_path,
            results_dir=args.results_dir,
            faiss_base_dir=args.faiss_base_dir,
            models_dir=args.models_dir,
            data_dir=args.data_dir,
            hybrid_top_k=args.hybrid_top_k,
            scoring_n_seeds=args.scoring_n_seeds,
            device=args.device,
            prefilter=args.prefilter,
        )
    else:
        defaults = _build_mode_config(
            args.mode, args.faiss_base_dir, args.models_dir, args.data_dir
        )
        output_csv = args.output or os.path.join(
            args.results_dir, f"full_experiment_ppo_{args.mode}.csv"
        )

        run_ppo_experiment(
            output_csv=output_csv,
            base_dir=args.base_dir,
            minigrid_ids_path=args.minigrid_ids_path,
            eval_seeds_path=args.eval_seeds_path,
            experiments=defaults["experiments"],
            policy_list=defaults["policy_list"],
            index_path=args.index_path or defaults["index_path"],
            metadata_path=args.metadata_path or defaults["metadata_path"],
            regressor_model_path=args.regressor_model_path
            or defaults["regressor_model_path"],
            regressor_variant=args.regressor_variant
            or defaults["regressor_variant"],
            canonical_states_path=args.canonical_states_path
            or defaults["canonical_states_path"],
            setup_name=defaults["setup_name"],
            hybrid_top_k=args.hybrid_top_k,
            scoring_n_seeds=args.scoring_n_seeds,
            device=args.device,
            prefilter=args.prefilter,
        )


if __name__ == "__main__":
    main()
