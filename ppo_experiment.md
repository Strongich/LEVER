# PPO Policy Composition Experiment Flow

This document describes the end-to-end flow of `ppo/full_experiment.py`, which evaluates three strategies for composing PPO sub-policies via Generalised Policy Improvement (GPI).

## Overview

The experiment answers: *given a complex multi-objective task described in natural language, can we compose single-objective PPO policies to outperform a monolithic policy trained from scratch on the full task?*

Three composition strategies are compared against a scratch baseline:

| Approach | Selection | Scoring |
|---|---|---|
| **Targeted** | Best-1 candidate per sub-query (by regressor rank) | None (trust the rank) |
| **Exhaustive** | All candidates above similarity threshold | SF embedding + regressor |
| **Hybrid** | Top-k candidates per sub-query (by regressor rank) | SF embedding + regressor |

## GPI (Generalised Policy Improvement) -- PPO Variant

The core composition mechanism differs from DQN. Since PPO is an actor-critic algorithm, GPI uses the **critic (value function)** to select which actor to trust:

```
V_best = max( V_1(s), V_2(s), ..., V_N(s) )
a* = actor_best(s)   (actor from the model with highest V(s))
```

At each step, every model's critic computes `V(s)` via `model.policy.predict_values(obs)`. The model with the highest state-value estimate is selected, and its actor's deterministic action is executed. This is implemented in `gpi_action_ppo()`.

**Key difference from DQN GPI**: DQN computes `Q(s,a)` for all actions across all models and takes `argmax_a max_i Q_i(s,a)`. PPO GPI instead selects the *best model* by V(s) and delegates action selection entirely to that model's actor network. This is because PPO's actor outputs a policy distribution, not per-action Q-values.

## Modes

The script runs in three modes, each using a different granularity of sub-policies in the FAISS index:

| Mode | Sub-policy granularity | FAISS index | Example sub-policies |
|---|---|---|---|
| **trivial** | Single-objective | `faiss_index_ppo/base/` | `path`, `gold`, `hazard`, `lever` |
| **double** | Two-objective | `faiss_index_ppo/pair/` | `path-gold`, `hazard-lever` |
| **triple** | Three-objective | `faiss_index_ppo/trip/` | `path-gold-hazard`, `lever` |

Each mode has its own FAISS index, metadata, regressor model, and canonical states. All three are run sequentially by `run_all_modes()`.

## Experiment Definitions

Each mode defines a set of experiments (setup + query + expected sub-query count):

**Trivial** (3 experiments):
- `path-gold` (2 sub-queries)
- `path-gold-hazard` (3 sub-queries)
- `path-gold-hazard-lever` (4 sub-queries)

**Double** (2 experiments):
- `path-gold-hazard` (2 sub-queries)
- `path-gold-hazard-lever` (2 sub-queries)

**Triple** (1 experiment):
- `path-gold-hazard-lever` (2 sub-queries)

## Step-by-Step Flow

### 1. Initialization

```
run_ppo_experiment()
```

Loads:
- **Eval seeds** from `eval_env_seeds.json` (100 seed integers)
- **Scoring seeds** = first N eval seeds (default 10), used for fast simulated rollouts during combo scoring
- **MiniGrid IDs** from `minigrid_ids.json` (object-type mappings for state encoding)
- **MiniGrid value map** built from the IDs (used by `state_to_vector()`)
- **Canonical states** from `.npy` file (reference state vectors for SF embedding)
- **PolicyRetriever** wrapping the FAISS index, metadata, and reward regressor

### 2. Per-Experiment Loop

For each experiment (e.g. `path-gold-hazard` in trivial mode):

#### 2a. Query Decomposition

```
sub_queries, decomp_time = decompose_query_with_retry(retriever, query, ...)
```

The natural-language query (e.g. *"Find the fastest exit, collect as much gold as possible, avoid hazards at all cost"*) is sent to an LLM which decomposes it into atomic sub-queries, one per objective. The LLM is constrained to return exactly `expected_count` sub-queries selected verbatim from a provided policy list. Retries up to 3 times if the count is wrong.

This function is imported from `tabular/full_experiment.py` (shared with DQN).

#### 2b. Pre-compute Environment States

```
state_cache = precompute_env_states(scoring_seeds, setup)
```

For the scoring seeds only (not all 100), creates a `LeverGridEnv` per seed, resets it, extracts the full grid state (`grid`, `agent_pos`, `agent_dir`), and closes the env. This cache enables simulated GPI rollouts without creating environments during combo scoring.

#### 2c. Scratch Baseline

```
scratch_model_path, scratch_time, scratch_reward_csv = get_scratch_info(base_dir, setup, prefilter)
```

Reads `episode_rewards.csv` from the scratch training run for this setup. Finds the best snapshot according to the prefilter:
- **v1**: `reward > 0` AND `reward_det > 0`, sorted descending by `[reward, reward_det]`.
- **v2**: same ranking, but no positivity filter.

Returns:
- **Model path**: `{base_dir}/{setup}/episodes/episode_{timesteps}/model.zip`
- **Scratch time**: `time_seconds` from the last CSV row (total training wall time)
- **Best reward_det**: deterministic eval reward from the CSV

#### 2d. Targeted Composition

```
tgt_models, tgt_names, t_targeted = targeted_composition(retriever, sub_queries, device)
```

For each sub-query:
1. Search FAISS for similar policies (filtered to `policy_seed="ppo"`)
2. Filter to similarity score > 0.7
3. Score all candidates with the regressor (`score_candidates()`)
4. Sort by `regressor_score` descending
5. Load the first candidate whose `.zip` file exists on disk

Returns one PPO model per sub-query. No combo scoring -- trusts the regressor ranking.

#### 2e. Exhaustive Composition

```
exh_models, exh_names, t_exhaustive, exh_pred = exhaustive_composition(...)
```

For each sub-query:
1. Search FAISS, filter to similarity > 0.7
2. Keep ALL candidates (no top-k pruning, no regressor pre-ranking)

Then for every combo in `itertools.product(*grouped)`:

1. **Load models** for the combo
2. **Simulated GPI rollout** on the cached scoring seeds (no env creation):
   - `obs_from_state()` replicates `LocalObsWrapper` + `ScaleObsWrapper` from the grid array
   - `gpi_action_ppo()` selects the action (critic-based actor selection)
   - `simulate_step()` advances the grid state (removes objects, checks termination)
   - `state_to_vector()` encodes both s and s' into feature vectors
   - Collects `(s_vec, s'_vec)` transitions
3. **Train a Successor Feature (SF) model** on those transitions (50 epochs, `SuccessorFeatureModelDeep`)
4. **Compute embedding** from SF model applied to canonical states
5. **Regressor prediction** on the embedding: `regressor.predict(embedding)`

The combo with the highest regressor prediction wins.

#### 2f. Hybrid Composition

```
hyb_models, hyb_names, t_hybrid, hyb_pred = hybrid_composition(...)
```

Same as exhaustive, except:
1. Candidates ARE pre-ranked by `score_candidates()` + regressor
2. Only top-k (default 3) candidates kept per sub-query

This reduces the combo count (e.g. from hundreds to `3^N`) while still doing SF-based scoring.

#### 2g. Evaluation

```
for seed in eval_seeds:  # all 100 seeds
```

For each seed, the scratch model and each composition's models are evaluated on the actual environment:

- **Scratch**: `evaluate_single_model()` -- creates a wrapped `LeverGridEnv`, runs `model.predict()` deterministically until done
- **Targeted/Exhaustive/Hybrid**: `evaluate_gpi_seed()` -- same env, but uses `gpi_action_ppo()` instead of a single model's `.predict()`

Both functions create a fresh environment per seed (with `render_mode="rgb_array"`), wrapped according to the configured `OBS_MODE` and `LOCAL_SIZE`.

### 3. CSV Output

One row per (experiment, seed) with columns:

```
setup, spec, seed, query,
scratch_reward, scratch_time_s,
targeted_reward, exhaustive_reward, hybrid_reward,
targeted_policies, exhaustive_policies, hybrid_policies,
exhaustive_predicted, hybrid_predicted,
decomp_time, targeted_time, exhaustive_time, hybrid_time
```

Output files: `results_ppo/full_experiment_ppo_{trivial,double,triple}.csv`

Note: The `spec` column is set to `"ppo8"` (matching the PPO policy seed).

## Environment Details

Default configuration uses the Grid-8 preset:

| Parameter | Value |
|---|---|
| Grid size | 8x8 |
| Balls (gold) | 6 |
| Walls | 4 |
| Lava (hazards) | 4 |
| Observation mode | local (7x7 patch) |
| Max steps | 128 |
| Action mode | cardinal (4 actions: right, down, left, up) |

Reward parameters: ball=15, key=20, exit=50 (200 for `path`), exit_with_key=100, lava=-1, step=-0.05.

Grid size and observation parameters can be overridden via `--grid-size`, `--obs-mode`, and `--local-size` CLI flags (e.g. `--grid-size 16 --obs-mode local --local-size 13` for 16x16 grids).

## Scoring Pipeline (Exhaustive/Hybrid) -- Detailed

This is the most complex part. For a single candidate combo:

```
combo models --> simulated GPI rollout on cached grid states
                          |
                    (s, s') transitions
                          |
                 train SF model (50 epochs)
                          |
                 compute embedding via canonical states
                          |
                 regressor.predict(embedding) --> score
```

The simulated rollout avoids creating real environments. Instead:
1. `obs_from_state()` extracts a local patch from the grid, transposes to channel-first, and applies scale factors `[25, 51, 127]` per channel
2. `simulate_step()` applies cardinal movement on the raw grid: walls block, balls/keys are removed on pickup, lava/goal terminate
3. `gpi_action_ppo()` evaluates V(s) per model's critic and uses the best model's actor
4. `state_to_vector()` encodes the full grid state into a flat feature vector for the SF model

This is much faster than creating `LeverGridEnv` instances, since environment creation involves Pygame/MiniGrid initialization overhead.

## pi2vec Preparation (ppo/pi2vec_preparation.py)

Before running the experiment, PPO policies must be prepared via `ppo/pi2vec_preparation.py`. This script:

1. **Selects snapshots** from `episode_rewards.csv` for each reward system:
   - 5 VDB snapshots at quantiles 0.2/0.4/0.6/0.8/1.0
   - 60% of remaining snapshots for regressor-only training
   - Prefilter `v1` requires positive rewards; `v2` uses all snapshots sorted by `reward_det`

2. **Encodes transitions** from `eval_transitions.pkl` via `state_to_vector()` with MiniGrid ID mappings

3. **Creates canonical states** (128 per setup) via reservoir sampling from encoded transitions

4. **Trains successor feature models** per snapshot, generates embeddings

5. **Populates FAISS VDBs** (base/pair/trip) with policy embeddings and metadata (including `policy_seed="ppo"` and `model_path`)

6. **Trains reward regressors** per setup to predict reward from SF embeddings

Output structure:
- `faiss_index_ppo/{base,pair,trip}/policy_ppo.index` + `metadata_ppo.pkl`
- `data_rl/canonical_states_ppo_{base,pair,trip}.npy`
- `models_ppo/reward_regressor_ppo_{base,pair,trip}.pkl`

## Key Differences from DQN Experiment

| Aspect | DQN (`dqn/full_experiment.py`) | PPO (`ppo/full_experiment.py`) |
|---|---|---|
| GPI mechanism | `max_i Q_i(s,a)` per action, then argmax | `max_i V_i(s)`, use best model's actor |
| Model loading | `DQN.load()` | `PPO.load()` |
| FAISS policy seed | `"dqn"` | `"ppo"` |
| Index paths | `faiss_index_dqn/` | `faiss_index_ppo/` |
| Model paths | `models_dqn/` | `models_ppo/` |
| Output CSVs | `full_experiment_dqn_{mode}.csv` | `full_experiment_ppo_{mode}.csv` |
| Spec column | `"dqn8"` | `"ppo8"` |
| Shared code | `decompose_query_with_retry()`, `score_candidates()` imported from `tabular/full_experiment.py` | Same |

## Key Functions Reference

| Function | Purpose |
|---|---|
| `get_ppo_model()` | Load + cache a `PPO.load()` model |
| `get_scratch_info()` | Find best snapshot path + training time from CSV |
| `decompose_query_with_retry()` | LLM query decomposition (imported from `tabular/full_experiment.py`) |
| `precompute_env_states()` | Cache initial grid states for scoring seeds |
| `obs_from_state()` | Grid array to CNN observation (no env needed) |
| `simulate_step()` | Advance grid state by one cardinal action |
| `gpi_action_ppo()` | GPI action selection via critic-based actor selection |
| `gpi_rollout_simulated()` | Full simulated rollout collecting transitions |
| `embedding_from_gpi_rollout()` | Transitions to SF embedding |
| `targeted_composition()` | Best-1 per sub-query |
| `exhaustive_composition()` | All combos, SF-scored |
| `hybrid_composition()` | Top-k per sub-query, SF-scored |
| `evaluate_single_model()` | Run one PPO on real env for one seed |
| `evaluate_gpi_seed()` | Run GPI on real env for one seed |
| `score_candidates()` | Regressor scoring (imported from `tabular/full_experiment.py`) |

## CLI Reference

```
uv run python ppo/full_experiment.py \
    --mode all \
    --grid-size 8 \
    --obs-mode local --local-size 7 \
    --prefilter v2 \
    --base-dir deeprl_runs_ppo_8/8 \
    --minigrid-ids-path deeprl_runs_ppo_8/minigrid_ids.json \
    --eval-seeds-path deeprl_runs_ppo_8/8/eval_env_seeds.json \
    --faiss-base-dir faiss_index_ppo_8 \
    --results-dir results_8_ppo
```

Key CLI flags:
| Flag | Default | Description |
|---|---|---|
| `--mode` | `all` | `trivial`, `double`, `triple`, or `all` |
| `--grid-size` | 8 | Updates `GRID_SIZE` and `MAX_STEPS` |
| `--obs-mode` | `local` | Observation mode for eval and simulated rollouts |
| `--local-size` | 7 | Patch side length for local obs mode |
| `--prefilter` | `v1` | Scratch baseline prefilter (`v1` or `v2`) |
| `--hybrid-top-k` | 3 | Top-k candidates per sub-query in hybrid mode |
| `--scoring-n-seeds` | 10 | Seeds used for simulated scoring rollouts |
| `--faiss-base-dir` | `faiss_index_ppo` | Base dir containing `base/`, `pair/`, `trip/` subdirs |
| `--models-dir` | `models_ppo` | Dir for regressor `.pkl` files |
| `--data-dir` | `data_rl` | Dir for canonical states `.npy` files |
| `--device` | `cuda` if available | Inference device |
