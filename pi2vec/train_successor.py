import json
import os
import pickle
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
    optimizer = Adam(model.parameters(), lr=1e-3)
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


def main(
    states_folder: str = "states_16",
    canonical_states: int = 64,
    run_dir: str | None = None,
    reward_systems: list[str] | None = None,
    regressor_data_path: str = "data/regressor_training_data.json",
    append_regressor_data: bool = False,
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
    regressor_training_data = {"policy_embedding": [], "reward": []}
    if append_regressor_data and os.path.exists(regressor_data_path):
        try:
            with open(regressor_data_path, "r") as f:
                existing = json.load(f)
            regressor_training_data["policy_embedding"].extend(
                existing.get("policy_embedding", [])
            )
            regressor_training_data["reward"].extend(existing.get("reward", []))
        except Exception as e:
            print(
                f"Warning: failed to load existing regressor data from {regressor_data_path}: {e}"
            )
    canonical_states_array = np.array(
        create_canonical_states(
            states_folder=states_folder,
            canonical_states=canonical_states,
            run_dir=run_dir,
            reward_systems=reward_systems,
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
        energy_j = getattr(r, "energy_j", None)
        time_s = getattr(r, "time_s", None)
        transitions = r.transitions
        policy_seed, policy_embedding = train_and_save_successor_model(
            policy_name, transitions, canonical_states_array
        )
        regressor_training_data["policy_embedding"].append(policy_embedding)
        regressor_training_data["reward"].append(reward)

        # Load Q-table from episode folder
        episode_id = r.episode_id
        seed_name = r.seed_name
        episode_id_int = int(episode_id)
        if run_dir:
            root_dir = run_dir
        else:
            root_dir = os.path.join(os.getcwd(), states_folder)

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
            "description": desc,
            "reward": reward,
            "policy_embedding": policy_embedding,
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
    policy_embeddings = []
    for embedding in regressor_training_data["policy_embedding"]:
        if isinstance(embedding, list):
            policy_embeddings.append(embedding)
        else:
            policy_embeddings.append(embedding.tolist())
    json_data = {
        "policy_embedding": policy_embeddings,
        "reward": regressor_training_data["reward"],
    }
    # Ensure data directory exists
    os.makedirs(os.path.dirname(regressor_data_path) or ".", exist_ok=True)
    with open(regressor_data_path, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
