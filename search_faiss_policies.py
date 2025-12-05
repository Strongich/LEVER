"""
Semantic search and policy retrieval for LEVER project using Faiss.

Multi-objective search: text description + optimization criteria
"""

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from faiss_utils.setup_faiss_vdb import FaissVectorDB
from pi2vec.train_regressor import load_model
from policy_reusability.my_work.init_gridworld import init_gridworld_rand

# Load environment variables
load_dotenv()


class QueryDecompositionResponse(BaseModel):
    """Pydantic model for query decomposition response."""

    queries: list[str]


class PolicyRetriever:
    """Retrieve and search policies from Faiss Vector Database."""

    def __init__(
        self,
        index_path="faiss_index/policy.index",
        metadata_path="faiss_index/metadata.pkl",
        application_name="Grid World",
        available_actions=None,
    ):
        """
        Initialize connection to Faiss.

        Args:
            index_path: Path to the Faiss index file
            metadata_path: Path to the metadata pickle file
            application_name: Name of the RL environment/application (default: "Grid World")
            available_actions: List of actions available in the environment
                             (default: ["up", "down", "left", "right"])
        """
        print("Loading Faiss vector database...")
        self.vdb = FaissVectorDB(index_path=index_path, metadata_path=metadata_path)
        self.vdb.load()
        print("✓ Database loaded")
        print()

        # Store environment configuration
        self.application_name = application_name
        self.available_actions = available_actions or [
            "right",
            "down",
            "twice right",
            "twice down",
            "diagonal right down",
        ]

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
            print("   Query decomposition will not be available")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=api_key)

        # Load regressor model for reward prediction
        print("Loading regressor model...")
        try:
            self.regressor_model = load_model("models/reward_regressor.pkl")
            print("✓ Regressor model loaded")
        except FileNotFoundError as e:
            print(f"⚠️  Warning: {e}")
            print("   Regressor scoring will not be available")
            self.regressor_model = None
        print()

    def decompose_query(self, query: str) -> list[str]:
        """
        Decompose a complex query into multiple sub-queries using LLM.

        This method uses OpenAI's GPT-5-nano to analyze a user query and break it
        down into multiple focused sub-queries that can be used for more effective
        retrieval.

        Args:
            query: The original user query to decompose

        Returns:
            list[str]: List of decomposed sub-queries

        Raises:
            RuntimeError: If OpenAI client is not initialized
        """
        if not self.openai_client:
            raise RuntimeError(
                "OpenAI client not initialized. Please set OPENAI_API_KEY in .env file"
            )

        # Format available actions for the prompt
        actions_str = ", ".join(self.available_actions)

        system_prompt = f"""You are a query decomposition assistant for Reinforcement Learning application.

Application Context:
- Environment: {self.application_name}
- Available Actions: {actions_str}

Your task is to decompose the user's query about a new and novel task into TRIVIAL tasks, where each sub-query describes only ONE single objective.

Important Guidelines:
- Break down complex tasks into the simplest possible single-objective components
- Each sub-query must represent ONE atomic objective only
- Avoid combining multiple objectives in a single sub-query
- Consider the RL environment context and available actions
- Each sub-query should be actionable for retrieving policies with a single clear goal

For example:
- If the query is "Navigate to target while collecting items", you must decompose it into:
  ["navigate to a target location", "collect items"]
  
- If the query is "Find shortest path quickly and efficiently", you might decompose it into:
  ["find shortest path to target", "minimize number of steps"]
  
- If the query is already a single objective --- return it as it is without any changes.

IMPORTANT: Do not return any other text than the list of sub-queries.
""".strip()

        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Decompose this query into sub-queries: {query}",
                    },
                ],
                response_format=QueryDecompositionResponse,
            )

            decomposed_queries = response.choices[0].message.parsed.queries

            print(f"🔍 Query decomposition ({len(decomposed_queries)} sub-queries):")
            for i, sub_query in enumerate(decomposed_queries, 1):
                print(f"   {i}. {sub_query}")
            print()

            return decomposed_queries

        except Exception as e:
            print(f"⚠️  Error during query decomposition: {e}")
            print("   Falling back to original query")
            return [query]

    async def search_decomposed_async(self, query_text, seed=None, filter_energy=False):
        """
        Decompose query and search for each sub-query asynchronously.
        For each decomposed query, returns the policy with highest regressor score.

        Args:
            query_text: Natural language query describing the desired policy
            seed: Policy seed to filter by (pre-filtering)
            filter_energy: If True, filter results to minimize energy consumption

        Returns:
            dict: {
                'original_query': str,
                'decomposed_queries': list[str],
                'results': list[dict] with 'sub_query', 'best_policy', and 'timing' for each
                'total_timing': dict
            }
        """
        start_time = time.time()

        # Step 1: Decompose the query
        decomposed_queries = self.decompose_query(query_text)
        decomposition_time = time.time() - start_time

        # Step 2: Search for each decomposed query in parallel
        search_start = time.time()

        # Create async tasks for each sub-query
        async def search_single_query(sub_query):
            """Search for a single sub-query and return best policy by regressor score."""
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                # Search with seed filtering
                result_dict, timing = await loop.run_in_executor(
                    executor,
                    lambda: self.vdb.search_similar_policies(
                        sub_query, policy_seed=seed
                    ),
                )

                # Check if search returned an error message
                if result_dict.get("message"):
                    return {
                        "sub_query": sub_query,
                        "best_policy": None,
                        "message": result_dict["message"],
                        "timing": timing,
                    }

                results = result_dict["results"]

                # Filter by cosine similarity threshold (> 0.7)
                similarity_threshold = 0.7
                similarity_filter_start = time.time()
                results = [
                    r for r in results if r.get("score", 0) > similarity_threshold
                ]
                similarity_filter_time = time.time() - similarity_filter_start
                timing["similarity_filter_time"] = similarity_filter_time
                timing["total_time"] += similarity_filter_time

                # If no results after similarity filtering, return None
                if not results:
                    return {
                        "sub_query": sub_query,
                        "best_policy": None,
                        "message": f"No policies found with cosine similarity > {similarity_threshold}",
                        "timing": timing,
                    }

                # Filter by energy if requested
                if filter_energy and results:
                    energy_filter_start = time.time()
                    results = [
                        r for r in results if r.get("energy_consumption") is not None
                    ]
                    if results:
                        # Sort by energy consumption (minimize)
                        results = sorted(
                            results,
                            key=lambda x: x.get("energy_consumption", float("inf")),
                        )
                    energy_filter_time = time.time() - energy_filter_start
                    timing["energy_filter_time"] = energy_filter_time
                    timing["total_time"] += energy_filter_time

                # Score each policy using regressor model
                if self.regressor_model and results:
                    regressor_start = time.time()
                    scored_policies = []
                    for policy in results:
                        # Get policy_embedding from metadata
                        policy_embedding = policy.get("policy_embedding")
                        if policy_embedding is not None:
                            # Convert to numpy array if needed
                            if isinstance(policy_embedding, list):
                                policy_embedding = np.array(policy_embedding)
                            # Predict reward using regressor
                            predicted_reward = self.regressor_model.predict(
                                policy_embedding.reshape(1, -1)
                            )[0]
                            policy["regressor_score"] = float(predicted_reward)
                            scored_policies.append(policy)

                    regressor_time = time.time() - regressor_start
                    timing["regressor_time"] = regressor_time
                    timing["total_time"] += regressor_time

                    # Argmax: get policy with highest regressor score
                    if scored_policies:
                        best_policy = max(
                            scored_policies,
                            key=lambda x: x.get("regressor_score", float("-inf")),
                        )
                    else:
                        best_policy = None
                else:
                    # If no regressor or no results, return None or first result
                    best_policy = results[0] if results else None

                return {
                    "sub_query": sub_query,
                    "best_policy": best_policy,
                    "message": None,
                    "timing": timing,
                }

        # Execute all searches in parallel
        search_results = await asyncio.gather(
            *[search_single_query(sq) for sq in decomposed_queries]
        )

        search_time = time.time() - search_start
        total_time = time.time() - start_time

        return {
            "original_query": query_text,
            "decomposed_queries": decomposed_queries,
            "seed": seed,
            "results": search_results,
            "total_timing": {
                "decomposition_time": decomposition_time,
                "search_time": search_time,
                "total_time": total_time,
            },
        }

    def search_with_decomposition(
        self, query_text, seed=None, filter_energy=False, show_all_metrics=False
    ):
        """
        Decompose query and search for each sub-query (synchronous wrapper).
        Returns the best policy per decomposed query based on regressor score.

        Args:
            query_text: Natural language query describing the desired policy
            seed: Policy seed to filter by (pre-filtering)
            filter_energy: If True, filter results to minimize energy consumption
            show_all_metrics: Whether to show all metadata fields

        Returns:
            Search results with decomposed queries and their best policies
        """
        # Run async search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.search_decomposed_async(
                    query_text, seed=seed, filter_energy=filter_energy
                )
            )
        finally:
            loop.close()

        # Print results
        print("=" * 80)
        print(f"Original Query: '{result['original_query']}'")
        if result.get("seed"):
            print(f"Seed Filter: {result['seed']}")
        print("=" * 80)
        print()

        # Show decomposed queries
        print(f"📋 Decomposed into {len(result['decomposed_queries'])} sub-queries:")
        for i, sq in enumerate(result["decomposed_queries"], 1):
            print(f"   {i}. {sq}")
        print()

        # Show timing
        timing = result["total_timing"]
        print(f"⏱️  Total time: {timing['total_time'] * 1000:.2f}ms")
        print(f"   - Decomposition: {timing['decomposition_time'] * 1000:.2f}ms")
        print(f"   - Parallel search: {timing['search_time'] * 1000:.2f}ms")
        print()
        print("=" * 80)
        print()

        dags_to_combine = []

        # Display best policy for each sub-query
        for i, search_result in enumerate(result["results"], 1):
            sub_query = search_result["sub_query"]
            best_policy = search_result["best_policy"]
            message = search_result.get("message")
            sub_timing = search_result["timing"]

            print(f"🔍 Sub-query {i}: '{sub_query}'")

            if message:
                print(f"   ⚠️  {message}")
            elif best_policy:
                # Show timing for this sub-query
                timing_parts = [
                    f"embedding: {sub_timing['embedding_time'] * 1000:.2f}ms"
                ]
                timing_parts.append(f"search: {sub_timing['search_time'] * 1000:.2f}ms")
                if "similarity_filter_time" in sub_timing:
                    timing_parts.append(
                        f"similarity_filter: {sub_timing['similarity_filter_time'] * 1000:.2f}ms"
                    )
                if "energy_filter_time" in sub_timing:
                    timing_parts.append(
                        f"energy_filter: {sub_timing['energy_filter_time'] * 1000:.2f}ms"
                    )
                if "regressor_time" in sub_timing:
                    timing_parts.append(
                        f"regressor: {sub_timing['regressor_time'] * 1000:.2f}ms"
                    )

                print(
                    f"   ⏱️  {sub_timing['total_time'] * 1000:.2f}ms ({', '.join(timing_parts)})"
                )
                print()

                # Display best policy
                print("   🏆 Best Policy (highest regressor score):")
                print(
                    f"      Name: {best_policy.get('policy_name', best_policy.get('name', 'N/A'))}"
                )
                if best_policy.get("policy_target"):
                    print(f"      Target: {best_policy.get('policy_target')}")
                print(f"      Cosine similarity: {best_policy.get('score', 0):.4f}")
                if "regressor_score" in best_policy:
                    print(
                        f"      Regressor Score (predicted reward): {best_policy['regressor_score']:.4f}"
                    )
                print(f"      Description: {best_policy.get('description', 'N/A')}")

                if show_all_metrics:
                    # Show comprehensive metrics
                    print(f"      Policy Seed: {best_policy.get('policy_seed', 'N/A')}")
                    print(
                        f"      Policy Target: {best_policy.get('policy_target', 'N/A')}"
                    )
                    print(
                        f"      RL Algorithm: {best_policy.get('rl_algorithm', 'N/A')}"
                    )
                    print(f"      Accuracy: {best_policy.get('accuracy', 0):.2%}")
                    print(
                        f"      Inference Time: {best_policy.get('inference_time_ms', 0):.2f}ms"
                    )
                    print(
                        f"      Energy: {best_policy.get('energy_consumption', 0):.2f}"
                    )
                    print(
                        f"      Memory: {best_policy.get('inference_memory_mb', 0):.2f}MB"
                    )
                    print(
                        f"      Model Size: {best_policy.get('model_size_mb', 0):.2f}MB"
                    )
                    print(
                        f"      Robustness: {best_policy.get('robustness_score', 0):.2%}"
                    )
                    if best_policy.get("q_table") is not None:
                        q_table = best_policy.get("q_table")
                        if isinstance(q_table, list):
                            q_table = np.array(q_table)
                        print(
                            f"      Q-table shape: {q_table.shape if hasattr(q_table, 'shape') else 'N/A'}"
                        )
                        print("      Q-table available: Yes")

                    if best_policy.get("dag") is not None:
                        dag = best_policy.get("dag")

                        print("      DAG type:", type(dag).__name__)

                        print("      DAG available: Yes")

                        dags_to_combine.append(dag)

                else:
                    # Show key metrics
                    if best_policy.get("energy_consumption") is not None:
                        print(
                            f"      Energy: {best_policy.get('energy_consumption', 0):.2f}"
                        )
                    if best_policy.get("reward") is not None:
                        print(f"      Reward: {best_policy.get('reward', 0):.4f}")
                    if best_policy.get("q_table") is not None:
                        q_table = best_policy.get("q_table")
                        if isinstance(q_table, list):
                            q_table = np.array(q_table)
                        print(
                            f"      Q-table shape: {q_table.shape if hasattr(q_table, 'shape') else 'N/A'}"
                        )
                        print("      Q-table available: Yes")

                    if best_policy.get("dag") is not None:
                        dag = best_policy.get("dag")

                        print("      DAG type:", type(dag).__name__)

                        print("      DAG available: Yes")

                        dags_to_combine.append(dag)
            else:
                print("   No policies found")

            print()
            print("-" * 80)
            print()

        from policy_reusability.pruning import run_pruning

        if len(dags_to_combine) == 2:
            combined_env = init_gridworld_rand(reward_system="combined", seed=seed)
            dag_1 = dags_to_combine[0]
            dag_2 = dags_to_combine[1]
            learning_rate = 0.1
            discount_factor = 0.99
            print("Start graph composition algorithm.")
            best_path, cumulative_reward_pruning, total_time, pruning_percentage = (
                run_pruning(
                    combined_env,
                    dag_1=dag_1,
                    dag_2=dag_2,
                    discount_factor=discount_factor,
                    learning_rate=learning_rate,
                )
            )

            print("=== Pruning Results ===")
            print(f"Best Path:               {best_path}")
            print(f"Cumulative Reward:       {cumulative_reward_pruning}")
            print(f"Total Time (seconds):    {total_time:.4f}")
            print(f"Pruning Percentage:      {pruning_percentage:.2f}%")
        else:
            print(
                f"Skipping graph composition: found {len(dags_to_combine)} DAG(s); need 2 with 'dag' metadata."
            )

        return result

    def optimize_results(self, results, optimize_by):
        """
        Re-rank results based on optimization criteria.

        Args:
            results: List of search results
            optimize_by: Optimization criterion ('inference_time', 'accuracy',
                        'energy', 'memory', 'model_size', 'robustness')

        Returns:
            Returns:
                tuple: (sorted_results, optimization_time)
        """

        start_time = time.time()

        # Define optimization strategies (minimize or maximize)
        minimize_metrics = [
            "inference_time",
            "energy",
            "memory",
            "model_size",
            "training_time",
        ]

        # Map user-friendly names to metadata keys
        metric_map = {
            "inference_time": "inference_time_ms",
            "accuracy": "accuracy",
            "energy": "energy_consumption",
            "memory": "inference_memory_mb",
            "model_size": "model_size_mb",
            "robustness": "robustness_score",
            "generalization": "generalization_score",
            "reward": "avg_episode_reward",
            "training_time": "training_time_hours",
        }

        metric_key = metric_map.get(optimize_by, optimize_by)

        # Sort based on metric
        if optimize_by in minimize_metrics:
            # Lower is better
            sorted_results = sorted(
                results, key=lambda x: x.get(metric_key, float("inf"))
            )
        else:
            # Higher is better
            sorted_results = sorted(
                results, key=lambda x: x.get(metric_key, float("-inf")), reverse=True
            )

        # Update ranks
        for i, result in enumerate(sorted_results, 1):
            result["rank"] = i

        optimization_time = time.time() - start_time

        return sorted_results, optimization_time

    def search(self, query_text, optimize_by=None, show_all_metrics=False):
        """
        Perform semantic search for policies with optional optimization.

        Args:
            query_text: Natural language query describing the desired policy
            optimize_by: Optional optimization criterion
            show_all_metrics: Whether to show all metadata fields

        Returns:
            Search results with policy information and timing
        """
        # Semantic search
        results, timing = self.vdb.search_similar_policies(query_text)

        # Apply optimization if requested
        optimization_time = 0
        if optimize_by:
            results, optimization_time = self.optimize_results(results, optimize_by)
            timing["optimization_time"] = optimization_time
            timing["total_time"] += optimization_time

        # Print results
        print("=" * 70)
        print(f"Query: '{query_text}'")
        if optimize_by:
            print(f"Optimized by: {optimize_by}")
        print("=" * 70)
        print(f"Found {len(results)} policies:")

        # Show timing breakdown
        timing_parts = [f"embedding: {timing['embedding_time'] * 1000:.2f}ms"]
        timing_parts.append(f"search: {timing['search_time'] * 1000:.2f}ms")
        if optimize_by:
            timing_parts.append(f"optimization: {optimization_time * 1000:.2f}ms")

        print(
            f"⏱️  Retrieval time: {timing['total_time'] * 1000:.2f}ms ({', '.join(timing_parts)})"
        )
        print()

        dags_to_combine = []

        # Display results
        for result in results:
            policy_name = result.get("policy_name", result.get("name", "N/A"))
            policy_target = result.get("policy_target", "N/A")
            target_str = f" (target: {policy_target})" if policy_target != "N/A" else ""
            print(f"{result['rank']}. {policy_name}{target_str}")
            print(f"   Cosine similarity: {result['score']:.4f}")
            description = result.get("description", result.get("objective", "N/A"))
            print(f"   Description: {description}")

            if show_all_metrics:
                # Show comprehensive metrics
                print(f"   RL Algorithm: {result.get('rl_algorithm', 'N/A')}")
                print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
                print(f"   Inference Time: {result.get('inference_time_ms', 0):.2f}ms")
                print(f"   Energy: {result.get('energy_consumption', 0):.2f}")
                print(f"   Memory: {result.get('inference_memory_mb', 0):.2f}MB")
                print(f"   Model Size: {result.get('model_size_mb', 0):.2f}MB")
                print(f"   Robustness: {result.get('robustness_score', 0):.2%}")
                if result.get("q_table") is not None:
                    q_table = result.get("q_table")
                    if isinstance(q_table, list):
                        q_table = np.array(q_table)
                    print(
                        f"   Q-table shape: {q_table.shape if hasattr(q_table, 'shape') else 'N/A'}"
                    )
                    print("   Q-table available: Yes")

                if result.get("dag") is not None:
                    dag = result.get("dag")

                    print("      DAG type:", type(dag).__name__)

                    print("      DAG available: Yes")

                dags_to_combine.append(dag)
            else:
                # Show key metrics based on optimization
                if optimize_by == "inference_time":
                    print(
                        f"   Inference Time: {result.get('inference_time_ms', 0):.2f}ms"
                    )
                elif optimize_by == "accuracy":
                    print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
                elif optimize_by == "energy":
                    print(f"   Energy: {result.get('energy_consumption', 0):.2f}")
                elif optimize_by == "memory":
                    print(f"   Memory: {result.get('inference_memory_mb', 0):.2f}MB")
                elif optimize_by == "robustness":
                    print(f"   Robustness: {result.get('robustness_score', 0):.2%}")

                # Show Q-table if available
                if result.get("q_table") is not None:
                    q_table = result.get("q_table")
                    if isinstance(q_table, list):
                        q_table = np.array(q_table)
                    print(
                        f"   Q-table shape: {q_table.shape if hasattr(q_table, 'shape') else 'N/A'}"
                    )
                    print("   Q-table available: Yes")

                if result.get("dag") is not None:
                    dag = result.get("dag")

                    print("      DAG type:", type(dag).__name__)

                    print("      DAG available: Yes")
                    dags_to_combine.append(dag)
            print()

        return results, timing


def main():
    parser = argparse.ArgumentParser(
        description="Search for similar policies using semantic similarity with regressor scoring"
    )
    parser.add_argument(
        "description",
        type=str,
        help="Natural language description of the new policy.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Policy seed to filter by (pre-filtering before search)",
    )
    parser.add_argument(
        "--filter-energy",
        action="store_true",
        help="Filter results to minimize energy consumption before regressor scoring",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all metadata fields for each policy",
    )

    args = parser.parse_args()

    # Initialize retriever
    try:
        retriever = PolicyRetriever(
            application_name="Grid World",
            available_actions=["move up", "move down", "move left", "move right"],
        )
    except Exception as e:
        print(f"Error loading Faiss database: {e}")
        print(
            "Please check the faiss_utils/setup_faiss_vdb.py module for more information."
        )
        return

    # Search with decomposition, seed filtering, and regressor scoring
    retriever.search_with_decomposition(
        args.description,
        seed=args.seed,
        filter_energy=args.filter_energy,
        show_all_metrics=True,  # args.show_all,
    )


if __name__ == "__main__":
    main()
