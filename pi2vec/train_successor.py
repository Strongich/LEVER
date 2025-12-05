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
):
    _, policy_seed, _ = policy_name.split("_")
    model = SuccessorFeatureModel()
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
    optimizer = Adam(model.parameters(), lr=3e-4)
    for _ in range(50):
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


def main():
    """Main function to train successor models and prepare training data."""
    from faiss_utils.setup_faiss_vdb import FaissVectorDB

    vdb = FaissVectorDB(
        index_path="faiss_index/policy.index", metadata_path="faiss_index/metadata.pkl"
    )
    regressor_training_data = {
        "policy_embedding": [],
        "reward": [],
    }
    canonical_states = np.array(create_canonical_states())
    processed_states = process_states()
    for r in tqdm(
        processed_states.itertuples(),
        total=len(processed_states),
        desc="Training successor models",
    ):
        policy_target = r.policy_target
        desc = (
            "Explore and collect as many gold pieces as possible."
            if policy_target == "gold"
            else "Find the exit cell as quickly as possible."
        )
        policy_name = r.policy_name
        reward = r.reward
        transitions = r.transitions
        policy_seed, policy_embedding = train_and_save_successor_model(
            policy_name, transitions, canonical_states
        )
        regressor_training_data["policy_embedding"].append(policy_embedding)
        regressor_training_data["reward"].append(reward)

        # Load Q-table from episode folder
        episode_id = r.episode_id
        seed_name = r.seed_name
        episode_str = f"episode_{int(episode_id):05d}"
        q_table_path = os.path.join(
            os.getcwd(),
            "states_f",
            policy_target,
            seed_name,
            "episodes",
            episode_str,
            "q_table.npy",
        )

        q_table = None
        if os.path.exists(q_table_path):
            try:
                q_table = np.load(q_table_path)
                # Convert to list for JSON serialization in metadata
                q_table = q_table.tolist()
            except Exception as e:
                print(f"Warning: Could not load Q-table from {q_table_path}: {e}")

        dag_path = os.path.join(
            os.getcwd(),
            "states_f",
            policy_target,
            seed_name,
            "episodes",
            episode_str,
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
            # example values
            "energy_consumption": round(np.random.uniform(0, 1), 3),
        }
        vdb.add_policy_from_kwargs(**faiss_entry)
    vdb.save()

    # Save regressor training data to JSON
    # Convert numpy arrays to lists for JSON serialization
    json_data = {
        "policy_embedding": [
            embedding.tolist()
            for embedding in regressor_training_data["policy_embedding"]
        ],
        "reward": regressor_training_data["reward"],
    }
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    # Save to JSON file
    json_path = "data/regressor_training_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
