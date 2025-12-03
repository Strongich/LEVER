"""
Setup Faiss Vector Database for LEVER project.

This script:
1. Creates a Faiss index for vector similarity search
2. Embeds policy descriptions using Alibaba-NLP/gte-multilingual-base
3. Stores policies with embeddings in Faiss
4. Saves metadata separately in a pickle file

Reference: https://github.com/facebookresearch/faiss
"""

import os
import pickle
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from policy_reusability.DAG import DAG


class FaissVectorDB:
    """Faiss Vector Database manager for policy storage and retrieval."""

    def __init__(self, index_path="faiss_index", metadata_path="faiss_metadata.pkl"):
        """Initialize Faiss vector database."""
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []  # List of policy info dicts
        self.policy_id_map = {}  # Map policy_id to index position

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            "Alibaba-NLP/gte-multilingual-base", trust_remote_code=True
        )
        self.embedding_dim = 768  # Dimension for gte-multilingual-base

    def create_index(self):
        """Create Faiss index for vector similarity search using cosine similarity."""
        # Create IndexFlatIP (Inner Product) for cosine similarity
        # We'll normalize vectors before adding them
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def embed_text(self, text):
        """Generate embedding for text using the model."""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def add_policy(self, policy_id, policy_info):
        """Add a policy to Faiss with its embedding."""
        # Generate embedding from description only
        # Everything else is stored as metadata for display
        embedding = self.embed_text(policy_info["description"])

        # Add to index
        self.index.add(embedding.reshape(1, -1))

        # Store all metadata (everything except description which is embedded)
        idx = len(self.metadata)
        metadata_entry = {
            "policy_id": policy_id,
            "name": policy_info["name"],
            "task": policy_info["task"],
            "description": policy_info["description"],
            "objective": policy_info["objective"],
            "reward_type": policy_info["reward_type"],
            # Performance metrics
            "accuracy": policy_info["accuracy"],
            "avg_episode_reward": policy_info["avg_episode_reward"],
            # Efficiency metrics
            "inference_time_ms": policy_info["inference_time_ms"],
            "inference_memory_mb": policy_info["inference_memory_mb"],
            "energy_consumption": policy_info["energy_consumption"],
            # Model properties
            "rl_algorithm": policy_info["rl_algorithm"],
            "model_size_mb": policy_info["model_size_mb"],
            "num_parameters": policy_info["num_parameters"],
            # Training info
            "training_time_hours": policy_info["training_time_hours"],
            "training_episodes": policy_info["training_episodes"],
            # Quality metrics
            "generalization_score": policy_info["generalization_score"],
            "robustness_score": policy_info["robustness_score"],
        }
        self.metadata.append(metadata_entry)
        self.policy_id_map[policy_id] = idx

    def _ensure_initialized(self):
        """
        Ensure index and metadata are initialized.
        If index doesn't exist, try to load from disk. If files don't exist, create new index.
        """
        # Check if index is already initialized
        if self.index is not None and len(self.metadata) > 0:
            return

        # Try to load from disk if files exist
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data["metadata"]
                    self.policy_id_map = data["policy_id_map"]
                return
            except Exception:
                # If loading fails, create new index
                pass

        # Create new index if it doesn't exist or loading failed
        if self.index is None:
            self.create_index()

        # Ensure metadata is initialized
        if not self.metadata:
            self.metadata = []
            self.policy_id_map = {}

    def add_policy_from_kwargs(self, **kwargs):
        """
        Add a policy to Faiss using **kwargs.

        The 'description' field will be embedded, all other fields will be stored as metadata.
        Automatically initializes index and metadata if they don't exist.

        Args:
            **kwargs: Policy information including 'description' field for embedding
                     and any other metadata fields (e.g., policy_seed, policy_target, etc.)

        Example:
            vdb.add_policy_from_kwargs(
                policy_id="policy_1",
                description="Explore and collect gold pieces",
                policy_seed=42,
                policy_target="gold",
                reward=85.3
            )
        """
        # Ensure index and metadata are initialized
        self._ensure_initialized()

        # Check if description exists
        if "description" not in kwargs:
            raise ValueError("'description' field is required for embedding")

        # Extract description for embedding
        description = kwargs["description"]
        embedding = self.embed_text(description)

        # Add to index
        self.index.add(embedding.reshape(1, -1))

        # Store all metadata (including description)
        idx = len(self.metadata)
        metadata_entry = kwargs.copy()  # Store all kwargs as metadata

        self.metadata.append(metadata_entry)

        # If policy_id is provided, add to policy_id_map
        if "policy_id" in kwargs:
            self.policy_id_map[kwargs["policy_id"]] = idx

    def save(self):
        """Save Faiss index and metadata to disk."""
        if self.index is None:
            raise ValueError("Index not initialized. Cannot save.")

        # Create directory for index if it doesn't exist
        index_dir = os.path.dirname(self.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)

        # Create directory for metadata if it doesn't exist
        metadata_dir = os.path.dirname(self.metadata_path)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)

        # Save Faiss index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {"metadata": self.metadata, "policy_id_map": self.policy_id_map}, f
            )

    def load(self):
        """Load Faiss index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Index file not found: {self.index_path}. Run setup first."
            )

        # Load Faiss index
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        with open(self.metadata_path, "rb") as f:
            print("Load pickle file")
            data = pickle.load(f)
            print("Loaded pickle file")
            self.metadata = data["metadata"]
            self.policy_id_map = data["policy_id_map"]

    def search_similar_policies(self, query_text, k=5, policy_seed=None):
        """
        Search for similar policies using semantic search.

        Args:
            query_text: Query string
            k: Number of results to return (None = all policies)
            policy_seed: Optional filter to only search among policies with this seed.
                        If None, searches all policies.

        Returns:
            Tuple of (result_dict, timing_dict) where:
            - result_dict: Dictionary with keys:
                - "results": List of dicts with policy info and similarity scores
                - "message": Optional message (None if results found, error message if no policies for seed)
                - "seed": The policy_seed that was searched (or None)
            - timing_dict: Dictionary with timing information
        """
        start_time = time.time()

        # Filter metadata by policy_seed if specified
        if policy_seed is not None:
            # Convert policy_seed to string for consistent comparison
            policy_seed_str = str(policy_seed)
            # Get indices of policies matching the seed
            filtered_indices = [
                i
                for i, meta in enumerate(self.metadata)
                if str(meta.get("policy_seed")) == policy_seed_str
            ]

            if not filtered_indices:
                # No policies found with this seed
                timing = {
                    "embedding_time": 0.0,
                    "search_time": 0.0,
                    "total_time": time.time() - start_time,
                }
                # Return empty results with a message indicating no policies found
                return {
                    "results": [],
                    "message": f"No policies found for seed {policy_seed}. This MDP state doesn't have policies to be reused.",
                    "seed": policy_seed,
                }, timing

            # Create a temporary filtered index for searching
            filtered_index = faiss.IndexFlatIP(self.embedding_dim)
            filtered_metadata_map = {}  # Map filtered index position to original index

            # Add only filtered vectors to the temporary index
            for orig_idx in filtered_indices:
                # Get the vector from the original index
                vector = self.index.reconstruct(orig_idx)
                filtered_pos = filtered_index.ntotal
                filtered_index.add(vector.reshape(1, -1))
                filtered_metadata_map[filtered_pos] = orig_idx

            # Set k to filtered size if not specified
            if k is None:
                k = len(filtered_indices)
            else:
                k = min(k, len(filtered_indices))
        else:
            # No filtering - use original index
            filtered_index = self.index
            filtered_metadata_map = None
            if k is None:
                k = len(self.metadata)

        # Generate query embedding (normalized)
        embedding_start = time.time()
        query_embedding = self.embed_text(query_text).reshape(1, -1)
        embedding_time = time.time() - embedding_start

        # Search in Faiss (either filtered or full index)
        search_start = time.time()
        # For IndexFlatIP with normalized vectors, scores are cosine similarities
        scores, indices = filtered_index.search(query_embedding, k)
        search_time = time.time() - search_start

        total_time = time.time() - start_time

        # Prepare results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0:  # Valid index
                # Map filtered index back to original if needed
                if filtered_metadata_map is not None:
                    orig_idx = filtered_metadata_map[idx]
                else:
                    orig_idx = idx

                result = self.metadata[orig_idx].copy()
                result["score"] = float(score)
                result["rank"] = i + 1
                results.append(result)

        # Add timing information
        timing = {
            "embedding_time": embedding_time,
            "search_time": search_time,
            "total_time": total_time,
        }

        # Return results in consistent format
        return {
            "results": results,
            "message": None,  # No special message when results are found
            "seed": policy_seed,
        }, timing

    def get_policy_info(self, policy_id):
        """Retrieve policy information by policy ID."""
        if policy_id not in self.policy_id_map:
            return None
        idx = self.policy_id_map[policy_id]
        return self.metadata[idx]

    def get_index_stats(self):
        """Get statistics about the index."""
        return {
            "total_policies": len(self.metadata),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.embedding_dim,
            "index_type": "IndexFlatIP (Cosine Similarity)",
        }


def main():
    """
    Main setup function.

    Note: This module is designed to be imported and used programmatically.
    Policies should be added using add_policy() or add_policy_from_kwargs() methods.
    """
    print("=" * 70)
    print("LEVER: RL Reusability using Vector Databases")
    print("Faiss Vector Database Module")
    print("=" * 70)
    print()
    print(
        "This module provides the FaissVectorDB class for vector database operations."
    )
    print("Import and use it in your scripts to add policies and perform searches.")
    print()
    print("Example usage:")
    print("  from faiss_utils.setup_faiss_vdb import FaissVectorDB")
    print("  vdb = FaissVectorDB()")
    print("  vdb.add_policy_from_kwargs(description='...', policy_id='...', ...)")
    print()


# Module can be imported without executing main()
# To run setup, call main() explicitly or use as a script: python -m faiss_utils.setup_faiss_vdb
