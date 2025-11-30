# LEVER: RL Policy Reusability Framework

A framework for semantic search and retrieval of Reinforcement Learning policies using vector databases and successor feature representations.

## Overview

This framework enables you to:

- **Store RL policies** with their embeddings, metadata, and Q-tables in a vector database
- **Search for similar policies** using natural language queries
- **Decompose complex queries** into sub-queries for multi-objective policy retrieval
- **Predict policy performance** using a regressor model trained on policy embeddings
- **Filter and rank policies** by similarity, energy consumption, and predicted rewards

The framework uses:
- **Faiss** for efficient vector similarity search
- **Successor Feature Models** ([π2vec](https://arxiv.org/abs/2306.09800)) for policy representation - a method for representing behaviors of black box policies as feature vectors using successor features
- **Semantic embeddings** (Alibaba-NLP/gte-multilingual-base) for query understanding
- **Reward regressor** for predicting policy performance

## Setup

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create --name lever python=3.12

# Activate the environment
conda activate lever

# Install dependencies
pip install -r requirements.txt
```

### Using Python Virtual Environment

```bash
# Create a virtual environment
python3.12 -m venv lever_env

# Activate the environment
# On Linux/Mac:
source lever_env/bin/activate
# On Windows:
lever_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Getting Started

### Step 1: Prepare the Framework

Run the preparation script to train all necessary models:

```bash
python pi2vec_preparation.py
```

**What `pi2vec_preparation.py` does:**

1. **Trains Successor Feature Models**: Creates policy embeddings by training successor feature models on state transitions from your training data
2. **Builds Faiss Vector Database**: Stores policy descriptions and embeddings in a searchable vector database
3. **Trains Reward Regressor**: Trains a model to predict policy rewards from policy embeddings
4. **Saves Training Data**: Prepares data for the regressor model

The script will:
- Process states from `states_f/` folder (both `gold` and `path` policies)
- **Data Processing**: Currently processes only 10% of available seeds from `states_f/`, and for each selected seed, creates 3 policy snapshots at different training stages: 20%, 60%, and 100% of training progress
- Train successor models for each policy
- Add policies to the Faiss database with metadata (including Q-tables)
- Train and save the reward regressor model

**After running, you'll see:**
- Available policy seeds in the CLI output
- Trained models in `psi_models/` folder (model names reflect the seeds used)
- Vector database in `faiss_index/` folder
- Regressor model in `models/reward_regressor.pkl`

**Note**: You can also check available seeds by looking at the `psi_models/` folder - the model filenames contain the seed information (e.g., `gold_0038_20.pth` indicates seed `0038`).

### Step 2: Search for Policies

Once the framework is prepared, you can search for policies using natural language queries:

```bash
python search_faiss_policies.py "your query here" [options]
```

**Basic Usage:**

```bash
# Simple search
python search_faiss_policies.py "Find the fastest exit"

# Search with seed filter (only policies from specific seed)
python search_faiss_policies.py "Find the fastest exit" --seed 0038

# Filter by energy consumption
python search_faiss_policies.py "collect gold efficiently" --seed 0038 --filter-energy

# Show all metadata fields
python search_faiss_policies.py "your query" --show-all
```

**Note**: Currently, the framework supports the following query combination:
- Multi-objective queries with seed filtering: `python search_faiss_policies.py "Find the fastest exit and collect as much gold as possible" --seed <seed>`

**How `search_faiss_policies.py` works:**

1. **Query Decomposition**: Uses LLM to break down complex queries into simpler sub-queries
2. **Semantic Search**: Searches the vector database for policies matching each sub-query
3. **Similarity Filtering**: Only considers policies with cosine similarity > 0.7
4. **Regressor Scoring**: Predicts reward for each policy using the trained regressor
5. **Best Policy Selection**: Returns the policy with the highest predicted reward for each sub-query

**Example Output:**

```
🔍 Sub-query 1: 'find the fastest exit'
   🏆 Best Policy (highest regressor score):
      Name: path_0038_100
      Target: path
      Cosine similarity: 0.8319
      Regressor Score (predicted reward): 135.5220
      Description: Find the exit cell as quickly as possible.
      Energy: 0.19
      Reward: 129.0000
      Q-table shape: (256, 5)
      Q-table available: Yes
```

**Command Line Options:**

- `description`: Natural language description of the desired policy (required)
- `--seed SEED`: Filter policies by seed (e.g., `--seed 0038`)
- `--filter-energy`: Filter results to minimize energy consumption
- `--show-all`: Display all metadata fields for each policy

## Resetting the Framework

If you want to use different training data or start fresh, use the reset script:

```bash
python reset_framework.py
```

**What `reset_framework.py` does:**

The script will delete:
1. All data from the `data/` folder
2. The trained regressor model (`models/reward_regressor.pkl`)
3. The Faiss vector database (`faiss_index/` folder)
4. All successor feature models (`psi_models/` folder)

**Warning**: This action permanently deletes all trained models and processed data. Make sure you have backups if needed.

After resetting, you can run `pi2vec_preparation.py` again with new training data.

## Project Structure

```
.
├── data/                          # Processed training data
│   ├── canonical_states.npy      # Canonical state representations
│   ├── processed_states.csv       # Policy metadata
│   └── regressor_training_data.json
├── faiss_index/                   # Vector database files
│   ├── policy.index               # Faiss index
│   └── metadata.pkl               # Policy metadata
├── models/                        # Trained models
│   └── reward_regressor.pkl       # Reward prediction model
├── psi_models/                    # Successor feature models
│   └── {policy_name}.pth         # Individual policy models
├── states_f/                      # Training data (gold and path policies)
│   ├── gold/
│   └── path/
├── faiss_utils/                   # Faiss database utilities
│   ├── setup_faiss_vdb.py         # Vector database setup
│   └── view_faiss_db.py           # View database contents
├── pi2vec/                        # Policy embedding utilities
│   ├── train_successor.py         # Train successor models
│   ├── train_regressor.py         # Train reward regressor
│   └── pi2vec_utils.py            # Utility functions
├── pi2vec_preparation.py          # Main preparation script
├── search_faiss_policies.py       # Policy search interface
├── reset_framework.py             # Reset framework script
└── README.md                      # This file
```

## Requirements

See `requirements.txt` for the full list of dependencies. Key packages include:

- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `sentence-transformers` - Semantic embeddings
- `torch` - Deep learning framework
- `numpy`, `pandas` - Data processing
- `openai` - Query decomposition (requires API key)
- `scikit-learn` - Regressor model

## Environment Variables

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_api_key_here
```

This is required for query decomposition functionality.

## Troubleshooting

**Issue**: "No policies found with cosine similarity > 0.7"
- **Solution**: The query might not match any policies well. Try rephrasing the query or check available policies using `python faiss_utils/view_faiss_db.py`

**Issue**: "Regressor model not found"
- **Solution**: Run `python pi2vec_preparation.py` to train the regressor model

**Issue**: "Database not found"
- **Solution**: Run `python pi2vec_preparation.py` to create the vector database

