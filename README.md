# LEVER: RL Policy Reusability Framework

Semantic search, retrieval, and composition of reinforcement learning policies using Faiss, π2vec successor features, and a lightweight reward regressor.

## Overview
- Store GridWorld policies with embeddings, metadata, Q-tables, and DAGs in a Faiss-backed vector DB.
- Search with natural language, optional seed filtering, cosine-similarity gating (>0.6), and regressor-based ranking.
- Decompose complex queries via OpenAI (optional) and automatically run graph composition when two DAGs are returned.
- Train successor-feature models and a linear regressor to predict reward from learned embeddings.

## Setup
```bash
# Conda (recommended)
conda create -n lever python=3.12
conda activate lever
pip install -r requirements.txt

# or venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file if you want LLM-based query decomposition; searches still run without it.

## Data Expectations
The preparation pipeline assumes a `states_<size>` folder (defaults to `states_16`) containing:
- `gold/seed_xxxx/episodes/episode_XXXXXX/{episode_states.npy,q_table.npy,dag.pkl}`
- `path/seed_xxxx/episodes/episode_XXXXXX/{episode_states.npy,q_table.npy,dag.pkl}`
- `episode_rewards.csv` inside each `seed_xxxx/` directory.

Both `gold` and `path` must contain the same seed IDs; all common seeds are processed. Energy consumption is currently randomized metadata.

## Prepare Models and Index
Run the one-shot prep script (uses `states_16` with 64 canonical states by default):
```bash
python pi2vec_preparation.py
```
What happens:
- Builds canonical states (`data/canonical_states_states_16.npy`) and processed transitions if missing.
- Trains successor-feature models for every seed × {20, 60, 100}% episode snapshot and writes embeddings/Q-tables/DAGs into `faiss_index/metadata.pkl` and `psi_models/`.
- Saves regressor training data to `data/regressor_training_data.json` and trains `models/reward_regressor.pkl` if not already present.
- Faiss index lives at `faiss_index/policy.index`; it is recreated/extended as policies are added.

To use a different states folder or canonical count, call `main` programmatically, e.g.:
```bash
python - <<'PY'
from pi2vec_preparation import main
main(states_folder="states_16_1", canonical_states=128)
PY
```

## Searching Policies
```bash
python search_faiss_policies.py "collect gold quickly" --seed 0038 --filter-energy
```
- Query is decomposed via OpenAI if a key is present; otherwise the original text is used.
- Results are filtered to cosine similarity > 0.6, optionally sorted by energy, then re-ranked by the reward regressor when available.
- The script currently always prints full metadata (the `--show-all` flag is ignored in code).
- If two results include DAG metadata, the script runs pruning-based composition and simulates the merged DAG in a combined GridWorld.

## Inspecting and Resetting
- View DB contents: `python -m faiss_utils.view_faiss_db`
- Reset everything (data, models, faiss_index, psi_models): `python reset_framework.py` and confirm the prompt.

## Project Structure
```
.
├── data/                      # canonical_states_*.npy, processed_states.csv, regressor_training_data.json
├── faiss_index/               # policy.index and metadata.pkl
├── models/                    # reward_regressor.pkl
├── psi_models/                # successor feature checkpoints
├── states_16/ (or states_16_1/) # gold/path seeds with episodes, q_tables, DAGs
├── faiss_utils/               # FaissVectorDB setup + viewer
├── pi2vec/                    # successor features, regressor, utilities
├── search_faiss_policies.py   # CLI search with decomposition/regressor/composition
├── pi2vec_preparation.py      # preparation entrypoint
├── reset_framework.py         # cleanup script
└── full_experiment.py         # targeted vs exhaustive composition experiment runner
```

## Troubleshooting
- Missing index/metadata or regressor: rerun `python pi2vec_preparation.py`.
- No results: ensure the seed exists in both `gold` and `path` folders and that cosine similarity passes 0.6; try relaxing the query.
- Query decomposition errors: set `OPENAI_API_KEY` or run without it; the script will fall back to the original query.
