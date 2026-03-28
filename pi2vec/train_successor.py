import json
import os
import pickle
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from pi2vec.pi2vec_utils import create_canonical_states, process_states
from pi2vec.psimodel import (
    StateTransitionDataset,
    SuccessorFeatureModel,
    SuccessorFeatureModelDeep,
    save_model,
    train_epoch,
)

# Import DAG and create module aliases for pickle compatibility
from policy_reusability.DAG import DAG

# Create module aliases for old import paths used in pickle files
# This allows pickle to find classes even if they were saved with old paths
if "DAG" not in sys.modules:
    import policy_reusability.DAG as DAG_module

    sys.modules["DAG"] = DAG_module

# Create aliases for env module (used by GridWorld and other classes)
if "env" not in sys.modules:
    import policy_reusability.env as env_module

    sys.modules["env"] = env_module

if "env.gridworld" not in sys.modules:
    import policy_reusability.env.gridworld as gridworld_module

    sys.modules["env.gridworld"] = gridworld_module


class DAGUnpickler(pickle.Unpickler):
    """Custom unpickler to handle old DAG and env import paths."""

    def find_class(self, module, name):
        # Map old module paths to new ones
        # Handle cases where objects were pickled with old import paths:
        # - module="DAG", name="DAG" (from DAG import DAG)
        # - module="env.DAG", name="DAG" (from env.DAG import DAG)
        # - module="env.gridworld", name="GridWorld" (from env.gridworld import GridWorld)
        # - module="policy_reusability.DAG", name="DAG" (current path)

        # Handle DAG class
        if name == "DAG":
            return DAG

        # Handle GridWorld from old env module
        if name == "GridWorld" and (
            module == "env.gridworld" or module.endswith(".env.gridworld")
        ):
            from policy_reusability.env.gridworld import GridWorld

            return GridWorld

        # For all other classes, try default behavior first
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError) as e:
            # If it's an env-related error, try to find the class in policy_reusability.env
            if "env" in str(e) or "env" in module:
                try:
                    # Try to import from policy_reusability.env
                    if "gridworld" in module:
                        from policy_reusability.env.gridworld import GridWorld

                        if name == "GridWorld":
                            return GridWorld
                except ImportError:
                    pass

            # If it's a DAG-related error, try to return DAG
            if "DAG" in str(e) or name == "DAG":
                return DAG

            raise


def train_and_save_successor_model(
    policy_name: str,
    transitions: List[Tuple[np.ndarray, np.ndarray]],
    canonical_states: np.ndarray,
    epochs: int = 50,
    show_progress: bool = False,
):
    parts = policy_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected policy_name format: {policy_name}")
    policy_seed = parts[-2]

    # Infer state_dim from the first transition
    if len(transitions) > 0 and len(transitions[0]) > 0:
        state_dim = transitions[0][0].shape[0]
    else:
        # Fallback: infer from canonical_states
        if len(canonical_states) > 0:
            state_dim = canonical_states[0].shape[0]
        else:
            raise ValueError(
                "Cannot infer state_dim: no transitions or canonical states available"
            )

    model = SuccessorFeatureModel(state_dim=state_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = StateTransitionDataset(transitions)
    # Use batch_size=8, but ensure we don't get batches with 1 sample
    # With 15-28 transitions, incomplete batches will have 2-7 samples
    # We'll skip any batch with 1 sample in train_epoch
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid fork issues with tokenizers
        drop_last=False,  # Keep all data, but skip batch_size=1 in training loop
    )
    optimizer = Adam(model.parameters(), lr=3e-5)
    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter, desc=f"Training SF model {policy_name}", leave=False
        )
    for _ in epoch_iter:
        train_epoch(model, dataloader, optimizer, device=device)
        save_model(model, policy_name)
    policy_embeddings = []
    model.eval()  # Set to eval mode for inference (BatchNorm requires batch_size > 1 in train mode)
    with torch.no_grad():
        for state in canonical_states:
            state = torch.from_numpy(state).to(device)
            # Add batch dimension: (state_dim) -> (1, state_dim)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            policy_embedding = model(state).detach().cpu().numpy()
            policy_embeddings.append(policy_embedding)
    policy_embedding = np.average(np.array(policy_embeddings), axis=0)
    # Flatten to 1D if needed (model output might be 2D with batch dimension)
    if policy_embedding.ndim > 1:
        policy_embedding = policy_embedding.flatten()
    torch.cuda.empty_cache()
    return policy_seed, policy_embedding


def _compute_policy_embedding(
    model, canonical_states: np.ndarray, device
) -> np.ndarray:
    if canonical_states is None or len(canonical_states) == 0:
        raise ValueError("Canonical states are empty; cannot compute embedding.")
    policy_embeddings = []
    model.eval()
    with torch.no_grad():
        for state in canonical_states:
            state = torch.from_numpy(state).to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            policy_embedding = model(state).detach().cpu().numpy()
            policy_embeddings.append(policy_embedding)
    policy_embedding = np.average(np.array(policy_embeddings), axis=0)
    if policy_embedding.ndim > 1:
        policy_embedding = policy_embedding.flatten()
    return policy_embedding


def train_and_save_successor_model_variants(
    policy_name: str,
    transitions: List[Tuple[np.ndarray, np.ndarray]],
    canonical_states_by_variant: dict[str, np.ndarray],
    epochs: int = 50,
    show_progress: bool = False,
    policy_seed: str | None = None,
    model_cls=SuccessorFeatureModel,
):
    parts = policy_name.split("_")
    if policy_seed is None:
        if len(parts) < 2:
            raise ValueError(f"Unexpected policy_name format: {policy_name}")
        policy_seed = parts[-2]

    if len(transitions) > 0 and len(transitions[0]) > 0:
        state_dim = transitions[0][0].shape[0]
    else:
        if canonical_states_by_variant:
            first_states = next(iter(canonical_states_by_variant.values()))
            state_dim = first_states[0].shape[0]
        else:
            raise ValueError(
                "Cannot infer state_dim: no transitions or canonical states available"
            )

    model = model_cls(state_dim=state_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = StateTransitionDataset(transitions)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = Adam(model.parameters(), lr=7e-4)
    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter, desc=f"Training SF model {policy_name}", leave=False
        )
    for _ in epoch_iter:
        train_epoch(model, dataloader, optimizer, device=device)
        save_model(model, policy_name)

    embeddings_by_variant = {}
    for variant, canonical_states in canonical_states_by_variant.items():
        embeddings_by_variant[variant] = _compute_policy_embedding(
            model, canonical_states, device
        )
    torch.cuda.empty_cache()
    return policy_seed, embeddings_by_variant


def train_and_save_successor_model_deep(
    policy_name: str,
    transitions: List[Tuple[np.ndarray, np.ndarray]],
    canonical_states: np.ndarray,
    epochs: int = 50,
    show_progress: bool = False,
    policy_seed: str = "deeprl",
):
    # Infer state_dim from the first transition
    if len(transitions) > 0 and len(transitions[0]) > 0:
        state_dim = transitions[0][0].shape[0]
    else:
        # Fallback: infer from canonical_states
        if len(canonical_states) > 0:
            state_dim = canonical_states[0].shape[0]
        else:
            raise ValueError(
                "Cannot infer state_dim: no transitions or canonical states available"
            )

    model = SuccessorFeatureModelDeep(state_dim=state_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = StateTransitionDataset(transitions)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = Adam(model.parameters(), lr=3e-5)
    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter, desc=f"Training SF model {policy_name}", leave=False
        )
    for _ in epoch_iter:
        train_epoch(model, dataloader, optimizer, device=device)
        save_model(model, policy_name)
    policy_embeddings = []
    model.eval()
    with torch.no_grad():
        for state in canonical_states:
            state = torch.from_numpy(state).to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            policy_embedding = model(state).detach().cpu().numpy()
            policy_embeddings.append(policy_embedding)
    policy_embedding = np.average(np.array(policy_embeddings), axis=0)
    if policy_embedding.ndim > 1:
        policy_embedding = policy_embedding.flatten()
    torch.cuda.empty_cache()
    return policy_seed, policy_embedding


def main(
    states_folder: str = "states_16",
    canonical_states: int = 64,
    run_dir: str | None = None,
    reward_systems: list[str] | None = None,
    regressor_data_path: str = "data/regressor_training_data.json",
    append_regressor_data: bool = False,
    collect_regressor_data: bool = True,
    index_path: str = "faiss_index/policy.index",
    metadata_path: str = "faiss_index/metadata.pkl",
):
    """
    Main function to train successor models and prepare training data.

    Args:
        states_folder: Name of the folder containing states (e.g., "states_16").
        canonical_states: Total number of canonical states to collect (default: 64)
        run_dir: Path to a state_runs/<spec> folder; if provided, overrides states_folder.
        reward_systems: Optional list of reward systems to include.
        regressor_data_path: Output path for regressor training data.
        append_regressor_data: If True, append to existing regressor data (when present).
        index_path: Output path for Faiss index.
        metadata_path: Output path for Faiss metadata.
    """
    from faiss_utils.setup_faiss_vdb import FaissVectorDB

    vdb = FaissVectorDB(index_path=index_path, metadata_path=metadata_path)
    regressor_training_data = None
    if collect_regressor_data:
        regressor_training_data = {
            "policy_embedding": [],
            "policy_embedding_base": [],
            "policy_embedding_pair": [],
            "policy_embedding_trip": [],
            "reward": [],
            "policy_target": [],
            "spec": [],
        }
        if append_regressor_data and os.path.exists(regressor_data_path):
            try:
                with open(regressor_data_path, "r") as f:
                    existing = json.load(f)
                existing_embeddings = existing.get("policy_embedding", [])
                existing_base = existing.get(
                    "policy_embedding_base", existing_embeddings
                )
                existing_pair = existing.get("policy_embedding_pair", existing_base)
                existing_trip = existing.get("policy_embedding_trip", existing_base)
                existing_rewards = existing.get("reward", [])
                existing_targets = existing.get("policy_target")
                if existing_targets is None:
                    existing_targets = [None] * len(existing_embeddings)
                existing_specs = existing.get("spec")
                if existing_specs is None:
                    existing_specs = [None] * len(existing_embeddings)
                regressor_training_data["policy_embedding"].extend(existing_embeddings)
                regressor_training_data["policy_embedding_base"].extend(existing_base)
                regressor_training_data["policy_embedding_pair"].extend(existing_pair)
                regressor_training_data["policy_embedding_trip"].extend(existing_trip)
                regressor_training_data["reward"].extend(existing_rewards)
                regressor_training_data["policy_target"].extend(existing_targets)
                regressor_training_data["spec"].extend(existing_specs)
            except Exception as e:
                print(
                    f"Warning: failed to load existing regressor data from {regressor_data_path}: {e}"
                )
    base_rewards = ["path", "gold", "hazard", "lever"]
    pair_rewards = base_rewards + ["path-gold", "hazard-lever"]
    trip_rewards = base_rewards + ["path-gold-hazard"]
    canonical_states_base = np.array(
        create_canonical_states(
            states_folder=states_folder,
            canonical_states=canonical_states,
            run_dir=run_dir,
            reward_systems=base_rewards,
            output_suffix="base",
        )
    )
    canonical_states_pair = np.array(
        create_canonical_states(
            states_folder=states_folder,
            canonical_states=canonical_states,
            run_dir=run_dir,
            reward_systems=pair_rewards,
            output_suffix="pair",
        )
    )
    canonical_states_trip = np.array(
        create_canonical_states(
            states_folder=states_folder,
            canonical_states=canonical_states,
            run_dir=run_dir,
            reward_systems=trip_rewards,
            output_suffix="trip",
        )
    )
    processed_states = process_states(
        states_folder=states_folder,
        run_dir=run_dir,
        reward_systems=reward_systems,
    )
    for r in tqdm(
        processed_states.itertuples(),
        total=len(processed_states),
        desc="Training successor models",
    ):
        policy_target = r.policy_target
        desc_map = {
            "path": "Find the shortest path to the exit.",
            "gold": "Collect as much gold as possible before exiting.",
            "lever": "Activate the lever before reaching the exit.",
            "hazard": "Reach the exit while avoiding hazards and staying away from them.",
            "hazard-lever": "Activate the lever and avoid hazards while reaching the exit.",
            "path-gold": "Find the fastest exit and collect as much gold as possible.",
            "path-gold-hazard": (
                "Find the fastest exit, collect as much gold as possible, and avoid hazards."
            ),
            "path-gold-hazard-lever": (
                "Find the fastest exit, collect as much gold as possible, avoid hazards, "
                "and activate the lever."
            ),
        }
        desc = desc_map.get(
            policy_target, f"Composite objective: {policy_target.replace('-', ', ')}."
        )
        policy_name = r.policy_name
        reward = r.reward
        policy_spec = getattr(r, "spec", None)
        if not policy_spec and isinstance(policy_name, str) and "_" in policy_name:
            policy_spec = policy_name.split("_", 1)[0]
        energy_j = getattr(r, "energy_j", None)
        time_s = getattr(r, "time_s", None)
        include_in_vdb = getattr(r, "include_in_vdb", True)
        transitions = r.transitions
        policy_seed, embeddings_by_variant = train_and_save_successor_model_variants(
            policy_name,
            transitions,
            {
                "base": canonical_states_base,
                "pair": canonical_states_pair,
                "trip": canonical_states_trip,
            },
        )
        policy_embedding = embeddings_by_variant["base"]
        # Load Q-table from episode folder
        episode_id = r.episode_id
        seed_name = r.seed_name
        episode_id_int = int(episode_id)
        if run_dir:
            root_dir = run_dir
        else:
            root_dir = os.path.join(os.getcwd(), states_folder)

        if collect_regressor_data:
            reward_value = float(reward)
            regressor_training_data["policy_embedding"].append(policy_embedding)
            regressor_training_data["policy_embedding_base"].append(
                embeddings_by_variant["base"]
            )
            regressor_training_data["policy_embedding_pair"].append(
                embeddings_by_variant["pair"]
            )
            regressor_training_data["policy_embedding_trip"].append(
                embeddings_by_variant["trip"]
            )
            regressor_training_data["reward"].append(reward_value)
            regressor_training_data["policy_target"].append(policy_target)
            regressor_training_data["spec"].append(policy_spec)

        if not include_in_vdb:
            continue

        episodes_dir = os.path.join(root_dir, policy_target, seed_name, "episodes")
        q_table_path = None
        for width in (6, 5, 4):
            episode_str = f"episode_{episode_id_int:0{width}d}"
            candidate = os.path.join(episodes_dir, episode_str, "q_table.npy")
            if os.path.exists(candidate):
                q_table_path = candidate
                break
        if q_table_path is None and os.path.isdir(episodes_dir):
            for name in os.listdir(episodes_dir):
                if not name.startswith("episode_"):
                    continue
                suffix = name.split("_", 1)[1]
                if suffix.isdigit() and int(suffix) == episode_id_int:
                    q_table_path = os.path.join(episodes_dir, name, "q_table.npy")
                    break

        q_table = None
        if q_table_path and os.path.exists(q_table_path):
            try:
                q_table = np.load(q_table_path)
                # Convert to list for JSON serialization in metadata
                q_table = q_table.tolist()
            except Exception as e:
                print(f"Warning: Could not load Q-table from {q_table_path}: {e}")

        if run_dir:
            dag_path = os.path.join(
                run_dir,
                policy_target,
                seed_name,
                "episodes",
                episode_str,  # episode_str already uses 6 digits
                "dag.pkl",
            )
        else:
            dag_path = os.path.join(
                os.getcwd(),
                states_folder,
                policy_target,
                seed_name,
                "episodes",
                episode_str,  # episode_str already uses 6 digits
                "dag.pkl",
            )

        dag = None
        if os.path.exists(dag_path):
            try:
                with open(dag_path, "rb") as f:
                    # Use custom unpickler to handle old import paths
                    dag = DAGUnpickler(f).load()
            except Exception as e:
                print(f"Warning: Could not load DAG from {dag_path}: {e}")

        faiss_entry = {
            "policy_target": policy_target,
            "policy_seed": policy_seed,
            "policy_name": policy_name,
            "spec": policy_spec,
            "description": desc,
            "reward": reward,
            "policy_embedding": policy_embedding,
            "policy_embedding_base": embeddings_by_variant["base"],
            "policy_embedding_pair": embeddings_by_variant["pair"],
            "policy_embedding_trip": embeddings_by_variant["trip"],
            "q_table": q_table,  # Q-table as list (or None if not found)
            "dag": dag,
            "energy_consumption": energy_j,
            "training_time_s": time_s,
        }
        vdb.add_policy_from_kwargs(**faiss_entry)

    # Only save if index was initialized (i.e., at least one policy was added)
    if vdb.index is not None:
        vdb.save()
    else:
        print("Warning: No policies were processed. Skipping save.")

    # Save regressor training data to JSON
    # Convert numpy arrays to lists for JSON serialization
    if collect_regressor_data:
        policy_embeddings = []
        for embedding in regressor_training_data["policy_embedding"]:
            if isinstance(embedding, list):
                policy_embeddings.append(embedding)
            else:
                policy_embeddings.append(embedding.tolist())
        json_data = {
            "policy_embedding": policy_embeddings,
            "policy_embedding_base": [
                emb.tolist() if not isinstance(emb, list) else emb
                for emb in regressor_training_data["policy_embedding_base"]
            ],
            "policy_embedding_pair": [
                emb.tolist() if not isinstance(emb, list) else emb
                for emb in regressor_training_data["policy_embedding_pair"]
            ],
            "policy_embedding_trip": [
                emb.tolist() if not isinstance(emb, list) else emb
                for emb in regressor_training_data["policy_embedding_trip"]
            ],
            "reward": regressor_training_data["reward"],
            "policy_target": regressor_training_data["policy_target"],
            "spec": regressor_training_data["spec"],
        }
        # Ensure data directory exists
        os.makedirs(os.path.dirname(regressor_data_path) or ".", exist_ok=True)
        with open(regressor_data_path, "w") as f:
            json.dump(json_data, f, indent=2)


def train_deeprl_successor_models(
    transitions_path: str = "data_rl/deeprl_transitions.pkl",
    canonical_states_path: str = "data_rl/deeprl_canonical_states.pkl",
    canonical_states_base_path: str = "data_rl/deeprl_canonical_states_base.pkl",
    canonical_states_pair_path: str = "data_rl/deeprl_canonical_states_pair.pkl",
    canonical_states_trip_path: str = "data_rl/deeprl_canonical_states_trip.pkl",
    regressor_data_path: str = "data_rl/regressor_training_data_deeprl.json",
    append_regressor_data: bool = True,
    index_path: str = "faiss_index/policy.index",
    metadata_path: str = "faiss_index/metadata.pkl",
    epochs: int = 50,
    show_progress: bool = False,
    policy_seed: str = "deeprl",
    spec: str = "deeprl16",
    canonical_count: int = 128,
):
    """
    Train successor models for deep RL policies and update VDB/regressor data.
    """
    from faiss_utils.setup_faiss_vdb import FaissVectorDB

    if not os.path.exists(transitions_path):
        raise FileNotFoundError(f"Missing transitions pickle: {transitions_path}")
    if not os.path.exists(canonical_states_path):
        raise FileNotFoundError(
            f"Missing canonical states pickle: {canonical_states_path}"
        )

    with open(transitions_path, "rb") as handle:
        dataset = pickle.load(handle)

    base_rewards = ["path", "gold", "hazard", "lever"]
    pair_rewards = base_rewards + ["path-gold", "hazard-lever"]
    trip_rewards = base_rewards + ["path-gold-hazard"]

    def sample_canonical_states(entries, allowed_targets, count):
        reservoir = []
        seen = 0
        for entry in entries:
            if entry.get("policy_target") not in allowed_targets:
                continue
            for s_vec, _ in entry.get("transitions", []):
                seen += 1
                if len(reservoir) < count:
                    reservoir.append(s_vec)
                else:
                    idx = random.randrange(seen)
                    if idx < count:
                        reservoir[idx] = s_vec
        return np.array(reservoir, dtype=np.float32)

    canonical_states_base = sample_canonical_states(
        dataset, base_rewards, canonical_count
    )
    canonical_states_pair = sample_canonical_states(
        dataset, pair_rewards, canonical_count
    )
    canonical_states_trip = sample_canonical_states(
        dataset, trip_rewards, canonical_count
    )

    if canonical_states_base.size == 0:
        raise ValueError("No canonical states available for base rewards.")
    if canonical_states_pair.size == 0:
        raise ValueError("No canonical states available for pair rewards.")
    if canonical_states_trip.size == 0:
        raise ValueError("No canonical states available for trip rewards.")

    if canonical_states_base.size:
        with open(canonical_states_base_path, "wb") as handle:
            pickle.dump(canonical_states_base, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if canonical_states_pair.size:
        with open(canonical_states_pair_path, "wb") as handle:
            pickle.dump(canonical_states_pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if canonical_states_trip.size:
        with open(canonical_states_trip_path, "wb") as handle:
            pickle.dump(canonical_states_trip, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Keep legacy path if provided, but prefer variant-specific sets.
    canonical_states = canonical_states_base
    if canonical_states.size == 0 and os.path.exists(canonical_states_path):
        try:
            with open(canonical_states_path, "rb") as handle:
                canonical_states = pickle.load(handle)
        except Exception:
            canonical_states = np.load(canonical_states_path)

    vdb = FaissVectorDB(index_path=index_path, metadata_path=metadata_path)

    regressor_training_data = {
        "policy_embedding": [],
        "policy_embedding_base": [],
        "policy_embedding_pair": [],
        "policy_embedding_trip": [],
        "reward": [],
        "policy_target": [],
        "spec": [],
    }
    if append_regressor_data and os.path.exists(regressor_data_path):
        try:
            with open(regressor_data_path, "r") as f:
                existing = json.load(f)
            existing_embeddings = existing.get("policy_embedding", [])
            existing_base = existing.get("policy_embedding_base", existing_embeddings)
            existing_pair = existing.get("policy_embedding_pair", existing_base)
            existing_trip = existing.get("policy_embedding_trip", existing_base)
            existing_rewards = existing.get("reward", [])
            existing_targets = existing.get("policy_target")
            if existing_targets is None:
                existing_targets = [None] * len(existing_embeddings)
            existing_specs = existing.get("spec")
            if existing_specs is None:
                existing_specs = [None] * len(existing_embeddings)
            regressor_training_data["policy_embedding"].extend(existing_embeddings)
            regressor_training_data["policy_embedding_base"].extend(existing_base)
            regressor_training_data["policy_embedding_pair"].extend(existing_pair)
            regressor_training_data["policy_embedding_trip"].extend(existing_trip)
            regressor_training_data["reward"].extend(existing_rewards)
            regressor_training_data["policy_target"].extend(existing_targets)
            regressor_training_data["spec"].extend(existing_specs)
        except Exception as e:
            print(
                f"Warning: failed to load existing regressor data from {regressor_data_path}: {e}"
            )

    desc_map = {
        "path": "Find the shortest path to the exit.",
        "gold": "Collect as much gold as possible before exiting.",
        "lever": "Activate the lever before reaching the exit.",
        "hazard": "Reach the exit while avoiding hazards and staying away from them.",
        "hazard-lever": "Activate the lever and avoid hazards while reaching the exit.",
        "path-gold": "Find the fastest exit and collect as much gold as possible.",
        "path-gold-hazard": (
            "Find the fastest exit, collect as much gold as possible, and avoid hazards."
        ),
        "path-gold-hazard-lever": (
            "Find the fastest exit, collect as much gold as possible, avoid hazards, "
            "and activate the lever."
        ),
    }

    for entry in dataset:
        policy_name = entry.get("model_name")
        transitions = entry.get("transitions", [])
        reward = entry.get("reward")
        policy_target = entry.get("policy_target")
        include_in_vdb = entry.get("include_in_vdb", True)
        model_path = entry.get("model_path")
        if not policy_target and isinstance(policy_name, str) and "_" in policy_name:
            policy_target = policy_name.rsplit("_", 1)[0]
        if not policy_target:
            policy_target = "unknown"
        policy_spec = spec

        desc = desc_map.get(
            policy_target, f"Composite objective: {policy_target.replace('-', ', ')}."
        )

        _, embeddings_by_variant = train_and_save_successor_model_variants(
            policy_name,
            transitions,
            {
                "base": canonical_states_base,
                "pair": canonical_states_pair,
                "trip": canonical_states_trip,
            },
            epochs=epochs,
            show_progress=show_progress,
            policy_seed=policy_seed,
            model_cls=SuccessorFeatureModelDeep,
        )
        policy_embedding = embeddings_by_variant["base"]

        regressor_training_data["policy_embedding"].append(policy_embedding)
        regressor_training_data["policy_embedding_base"].append(
            embeddings_by_variant["base"]
        )
        regressor_training_data["policy_embedding_pair"].append(
            embeddings_by_variant["pair"]
        )
        regressor_training_data["policy_embedding_trip"].append(
            embeddings_by_variant["trip"]
        )
        regressor_training_data["reward"].append(float(reward))
        regressor_training_data["policy_target"].append(policy_target)
        regressor_training_data["spec"].append(policy_spec)

        if include_in_vdb:
            faiss_entry = {
                "policy_target": policy_target,
                "policy_seed": policy_seed,
                "policy_name": policy_name,
                "spec": spec,
                "description": desc,
                "reward": reward,
                "policy_embedding": policy_embedding,
                "policy_embedding_base": embeddings_by_variant["base"],
                "policy_embedding_pair": embeddings_by_variant["pair"],
                "policy_embedding_trip": embeddings_by_variant["trip"],
                "q_table": None,
                "dag": None,
                "energy_consumption": None,
                "training_time_s": None,
                "model_path": model_path,
            }
            vdb.add_policy_from_kwargs(**faiss_entry)

    if vdb.index is not None:
        vdb.save()
    else:
        print("Warning: No policies were processed. Skipping save.")

    policy_embeddings = []
    for embedding in regressor_training_data["policy_embedding"]:
        if isinstance(embedding, list):
            policy_embeddings.append(embedding)
        else:
            policy_embeddings.append(embedding.tolist())
    json_data = {
        "policy_embedding": policy_embeddings,
        "policy_embedding_base": [
            emb.tolist() if not isinstance(emb, list) else emb
            for emb in regressor_training_data["policy_embedding_base"]
        ],
        "policy_embedding_pair": [
            emb.tolist() if not isinstance(emb, list) else emb
            for emb in regressor_training_data["policy_embedding_pair"]
        ],
        "policy_embedding_trip": [
            emb.tolist() if not isinstance(emb, list) else emb
            for emb in regressor_training_data["policy_embedding_trip"]
        ],
        "reward": regressor_training_data["reward"],
        "policy_target": regressor_training_data["policy_target"],
        "spec": regressor_training_data["spec"],
    }
    os.makedirs(os.path.dirname(regressor_data_path) or ".", exist_ok=True)
    with open(regressor_data_path, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
