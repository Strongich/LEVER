import json
import os
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
        faiss_entry = {
            "policy_target": policy_target,
            "policy_seed": policy_seed,
            "policy_name": policy_name,
            "description": desc,
            "reward": reward,
            "policy_embedding": policy_embedding,
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
