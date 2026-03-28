SHELL := /bin/bash

.DEFAULT_GOAL := help

UV ?= uv
PYTHON := $(UV) run python
HYBRID_TOP_K ?= 3

STATES_8_0 := states_8_0
FAISS_8_0 := faiss_index_8_0
DATA_8_0 := data_8_0
MODELS_8_0 := models_8_0
RESULTS_8_0 := results_8_0
COMPARISONS_8_0 := comparisons_8_0

STATES_8_99 := states_8_99
FAISS_8_99 := faiss_index_8_99
DATA_8_99 := data_8_99
MODELS_8_99 := models_8_99
RESULTS_8_99 := results_8_99
COMPARISONS_8_99 := comparisons_8_99

STATES_16_0 := states_16_0
FAISS_16_0 := faiss_index_16_0
DATA_16_0 := data_16_0
MODELS_16_0 := models_16_0
RESULTS_16_0 := results_16_0
COMPARISONS_16_0 := comparisons_16_0

STATES_16_99 := states_16_99
FAISS_16_99 := faiss_index_16_99
DATA_16_99 := data_16_99
MODELS_16_99 := models_16_99
RESULTS_16_99 := results_16_99
COMPARISONS_16_99 := comparisons_16_99

DQN_RUNS_8 := deeprl_runs_dqn_8
DQN_BASE_DIR_8 := $(DQN_RUNS_8)/8
DQN_FAISS := faiss_index_dqn
DQN_DATA := data_rl
DQN_MODELS := models_dqn
DQN_RESULTS := results_dqn

PPO_RUNS_8 := deeprl_runs_ppo_8
PPO_BASE_DIR_8 := $(PPO_RUNS_8)/8
PPO_FAISS := faiss_index_ppo_8
PPO_DATA := data_rl
PPO_MODELS := models_ppo
PPO_RESULTS := results_8_ppo

.PHONY: help sync lock \
	states-8-0 states-8-99 states-16-0 states-16-99 \
	prep-8-0 prep-8-99 prep-16-0 prep-16-99 \
	exp-8-0 exp-8-99 exp-16-0 exp-16-99 \
	plots-8-0 plots-8-99 plots-16-0 plots-16-99 \
	sweep-8-0 sweep-8-99 sweep-16-0 sweep-16-99 \
	repro-8-0 repro-8-99 repro-16-0 repro-16-99 repro-all \
	dqn-train-8 dqn-prep-8 dqn-exp-8 dqn-repro-8 \
	ppo-train-8 ppo-prep-8 ppo-exp-8 ppo-repro-8

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

sync: ## Create or update the uv-managed virtual environment
	$(UV) sync

lock: ## Refresh uv.lock from pyproject.toml
	$(UV) lock

states-8-0: ## Train 8x8 SARSA policies with gamma=0 into states_8_0
	$(PYTHON) policy_reusability/data_generation/tabular/generate_states_batch.py \
		--output-root $(STATES_8_0) \
		--spec-set grid8 \
		--gamma 0

states-8-99: ## Train 8x8 SARSA policies with gamma=0.99 into states_8_99
	$(PYTHON) policy_reusability/data_generation/tabular/generate_states_batch.py \
		--output-root $(STATES_8_99) \
		--spec-set grid8 \
		--gamma 0.99

states-16-0: ## Train 16x16 SARSA policies with gamma=0 into states_16_0
	$(PYTHON) policy_reusability/data_generation/tabular/generate_states_batch.py \
		--output-root $(STATES_16_0) \
		--spec-set grid16_scaled \
		--gamma 0

states-16-99: ## Train 16x16 SARSA policies with gamma=0.99 into states_16_99
	$(PYTHON) policy_reusability/data_generation/tabular/generate_states_batch.py \
		--output-root $(STATES_16_99) \
		--spec-set grid16_scaled \
		--gamma 0.99

prep-8-0: ## Build pi2vec assets for states_8_0
	$(PYTHON) tabular/pi2vec_preparation.py \
		--base-dir $(STATES_8_0) \
		--index-path $(FAISS_8_0)/policy.index \
		--metadata-path $(FAISS_8_0)/metadata.pkl \
		--regressor-data-path $(DATA_8_0)/regressor_training_data.json \
		--regressor-base-path '$(MODELS_8_0)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_8_0)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_8_0)/{spec}/reward_regressor_trip.pkl' \
		--include-combined-rewards \
		--overwrite \
		--split-regressor-by-spec

prep-8-99: ## Build pi2vec assets for states_8_99
	$(PYTHON) tabular/pi2vec_preparation.py \
		--base-dir $(STATES_8_99) \
		--index-path $(FAISS_8_99)/policy.index \
		--metadata-path $(FAISS_8_99)/metadata.pkl \
		--regressor-data-path $(DATA_8_99)/regressor_training_data.json \
		--regressor-base-path '$(MODELS_8_99)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_8_99)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_8_99)/{spec}/reward_regressor_trip.pkl' \
		--include-combined-rewards \
		--overwrite \
		--split-regressor-by-spec

prep-16-0: ## Build pi2vec assets for states_16_0
	$(PYTHON) tabular/pi2vec_preparation.py \
		--base-dir $(STATES_16_0) \
		--index-path $(FAISS_16_0)/policy.index \
		--metadata-path $(FAISS_16_0)/metadata.pkl \
		--regressor-data-path $(DATA_16_0)/regressor_training_data.json \
		--regressor-base-path '$(MODELS_16_0)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_16_0)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_16_0)/{spec}/reward_regressor_trip.pkl' \
		--include-combined-rewards \
		--overwrite \
		--split-regressor-by-spec

prep-16-99: ## Build pi2vec assets for states_16_99
	$(PYTHON) tabular/pi2vec_preparation.py \
		--base-dir $(STATES_16_99) \
		--index-path $(FAISS_16_99)/policy.index \
		--metadata-path $(FAISS_16_99)/metadata.pkl \
		--regressor-data-path $(DATA_16_99)/regressor_training_data.json \
		--regressor-base-path '$(MODELS_16_99)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_16_99)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_16_99)/{spec}/reward_regressor_trip.pkl' \
		--include-combined-rewards \
		--overwrite \
		--split-regressor-by-spec

exp-8-0: ## Run the 8x8 gamma=0 composition experiments
	$(PYTHON) tabular/full_experiment.py \
		--loop-specs \
		--states-folder $(STATES_8_0) \
		--results-dir $(RESULTS_8_0) \
		--index-path $(FAISS_8_0)/policy.index \
		--metadata-path $(FAISS_8_0)/metadata.pkl \
		--hybrid-top-k $(HYBRID_TOP_K) \
		--regressor-base-path '$(MODELS_8_0)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_8_0)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_8_0)/{spec}/reward_regressor_trip.pkl'

exp-8-99: ## Run the 8x8 gamma=0.99 composition experiments
	$(PYTHON) tabular/full_experiment.py \
		--loop-specs \
		--states-folder $(STATES_8_99) \
		--results-dir $(RESULTS_8_99) \
		--index-path $(FAISS_8_99)/policy.index \
		--metadata-path $(FAISS_8_99)/metadata.pkl \
		--hybrid-top-k $(HYBRID_TOP_K) \
		--regressor-base-path '$(MODELS_8_99)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_8_99)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_8_99)/{spec}/reward_regressor_trip.pkl'

exp-16-0: ## Run the 16x16 gamma=0 composition experiments
	$(PYTHON) tabular/full_experiment.py \
		--loop-specs \
		--states-folder $(STATES_16_0) \
		--results-dir $(RESULTS_16_0) \
		--index-path $(FAISS_16_0)/policy.index \
		--metadata-path $(FAISS_16_0)/metadata.pkl \
		--hybrid-top-k $(HYBRID_TOP_K) \
		--regressor-base-path '$(MODELS_16_0)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_16_0)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_16_0)/{spec}/reward_regressor_trip.pkl'

exp-16-99: ## Run the 16x16 gamma=0.99 composition experiments
	$(PYTHON) tabular/full_experiment.py \
		--loop-specs \
		--states-folder $(STATES_16_99) \
		--results-dir $(RESULTS_16_99) \
		--index-path $(FAISS_16_99)/policy.index \
		--metadata-path $(FAISS_16_99)/metadata.pkl \
		--hybrid-top-k $(HYBRID_TOP_K) \
		--regressor-base-path '$(MODELS_16_99)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_16_99)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_16_99)/{spec}/reward_regressor_trip.pkl'

plots-8-0: ## Generate comparison plots for results_8_0
	mkdir -p $(COMPARISONS_8_0)
	$(PYTHON) plots/compare_compositions_average.py \
		--results-dir $(RESULTS_8_0) \
		--output $(COMPARISONS_8_0)/average_results.png
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_8_0) \
		--mode trivial \
		--output-dir $(COMPARISONS_8_0)/trivial
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_8_0) \
		--mode double \
		--output-dir $(COMPARISONS_8_0)/double
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_8_0) \
		--mode triple \
		--output-dir $(COMPARISONS_8_0)/triple

plots-8-99: ## Generate comparison plots for results_8_99
	mkdir -p $(COMPARISONS_8_99)
	$(PYTHON) plots/compare_compositions_average.py \
		--results-dir $(RESULTS_8_99) \
		--output $(COMPARISONS_8_99)/average_results.png
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_8_99) \
		--mode trivial \
		--output-dir $(COMPARISONS_8_99)/trivial
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_8_99) \
		--mode double \
		--output-dir $(COMPARISONS_8_99)/double
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_8_99) \
		--mode triple \
		--output-dir $(COMPARISONS_8_99)/triple

plots-16-0: ## Generate comparison plots for results_16_0
	mkdir -p $(COMPARISONS_16_0)
	$(PYTHON) plots/compare_compositions_average.py \
		--results-dir $(RESULTS_16_0) \
		--output $(COMPARISONS_16_0)/average_results.png
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_16_0) \
		--mode trivial \
		--output-dir $(COMPARISONS_16_0)/trivial
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_16_0) \
		--mode double \
		--output-dir $(COMPARISONS_16_0)/double
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_16_0) \
		--mode triple \
		--output-dir $(COMPARISONS_16_0)/triple

plots-16-99: ## Generate comparison plots for results_16_99
	mkdir -p $(COMPARISONS_16_99)
	$(PYTHON) plots/compare_compositions_average.py \
		--results-dir $(RESULTS_16_99) \
		--output $(COMPARISONS_16_99)/average_results.png
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_16_99) \
		--mode trivial \
		--output-dir $(COMPARISONS_16_99)/trivial
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_16_99) \
		--mode double \
		--output-dir $(COMPARISONS_16_99)/double
	$(PYTHON) plots/compare_compositions.py \
		--results-dir $(RESULTS_16_99) \
		--mode triple \
		--output-dir $(COMPARISONS_16_99)/triple

sweep-8-0: ## Run the hybrid top-k sweep for states_8_0
	$(PYTHON) hybrid_k_sweep.py \
		--state-runs-dir $(STATES_8_0) \
		--output $(RESULTS_8_0)/hybrid_k_sweep.csv \
		--index-path $(FAISS_8_0)/policy.index \
		--metadata-path $(FAISS_8_0)/metadata.pkl \
		--regressor-base-path '$(MODELS_8_0)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_8_0)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_8_0)/{spec}/reward_regressor_trip.pkl'
	$(PYTHON) plots/hybrid_k_sweep_plot.py \
		--input-csv $(RESULTS_8_0)/hybrid_k_sweep.csv \
		--output plots/hybrid_k_sweep_8_0.png

sweep-8-99: ## Run the hybrid top-k sweep for states_8_99
	$(PYTHON) hybrid_k_sweep.py \
		--state-runs-dir $(STATES_8_99) \
		--output $(RESULTS_8_99)/hybrid_k_sweep.csv \
		--index-path $(FAISS_8_99)/policy.index \
		--metadata-path $(FAISS_8_99)/metadata.pkl \
		--regressor-base-path '$(MODELS_8_99)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_8_99)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_8_99)/{spec}/reward_regressor_trip.pkl'
	$(PYTHON) plots/hybrid_k_sweep_plot.py \
		--input-csv $(RESULTS_8_99)/hybrid_k_sweep.csv \
		--output plots/hybrid_k_sweep_8_99.png

sweep-16-0: ## Run the hybrid top-k sweep for states_16_0
	$(PYTHON) hybrid_k_sweep.py \
		--state-runs-dir $(STATES_16_0) \
		--output $(RESULTS_16_0)/hybrid_k_sweep.csv \
		--index-path $(FAISS_16_0)/policy.index \
		--metadata-path $(FAISS_16_0)/metadata.pkl \
		--regressor-base-path '$(MODELS_16_0)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_16_0)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_16_0)/{spec}/reward_regressor_trip.pkl'
	$(PYTHON) plots/hybrid_k_sweep_plot.py \
		--input-csv $(RESULTS_16_0)/hybrid_k_sweep.csv \
		--output plots/hybrid_k_sweep_16_0.png

sweep-16-99: ## Run the hybrid top-k sweep for states_16_99
	$(PYTHON) hybrid_k_sweep.py \
		--state-runs-dir $(STATES_16_99) \
		--output $(RESULTS_16_99)/hybrid_k_sweep.csv \
		--index-path $(FAISS_16_99)/policy.index \
		--metadata-path $(FAISS_16_99)/metadata.pkl \
		--regressor-base-path '$(MODELS_16_99)/{spec}/reward_regressor_base.pkl' \
		--regressor-pair-path '$(MODELS_16_99)/{spec}/reward_regressor_pair.pkl' \
		--regressor-trip-path '$(MODELS_16_99)/{spec}/reward_regressor_trip.pkl'
	$(PYTHON) plots/hybrid_k_sweep_plot.py \
		--input-csv $(RESULTS_16_99)/hybrid_k_sweep.csv \
		--output plots/hybrid_k_sweep_16_99.png

repro-8-0: states-8-0 prep-8-0 exp-8-0 plots-8-0 ## Run the full 8x8 gamma=0 tabular pipeline

repro-8-99: states-8-99 prep-8-99 exp-8-99 plots-8-99 ## Run the full 8x8 gamma=0.99 tabular pipeline

repro-16-0: states-16-0 prep-16-0 exp-16-0 plots-16-0 ## Run the full 16x16 gamma=0 tabular pipeline

repro-16-99: states-16-99 prep-16-99 exp-16-99 plots-16-99 ## Run the full 16x16 gamma=0.99 tabular pipeline

repro-all: repro-8-0 repro-8-99 repro-16-0 repro-16-99 ## Run all tabular pipelines

dqn-train-8: ## Train the 8x8 DQN library with the documented experiment settings
	$(PYTHON) policy_reusability/data_generation/deeprl/train_dqn.py \
		--output-root $(DQN_RUNS_8) \
		--grid-preset grid8 \
		--timesteps 2000000 \
		--obs-mode local --local-size 7 \
		--snapshot-interval 0 --snapshot-steps 50000 \
		--n-envs 4 \
		--learning-starts 10000 \
		--exploration-fraction 0.25 \
		--exploration-final-eps 0.05 \
		--train-freq 4 --gradient-steps 1 \
		--target-update-interval 10000 --tau 1.0 --batch-size 64 \
		--loss-plot \
		--overwrite \
		--all-rewards

dqn-prep-8: ## Build pi2vec assets for the 8x8 DQN library
	$(PYTHON) dqn/pi2vec_preparation.py \
		--base-dir $(DQN_BASE_DIR_8) \
		--minigrid-ids-path $(DQN_RUNS_8)/minigrid_ids.json \
		--index-dir $(DQN_FAISS) \
		--data-dir $(DQN_DATA) \
		--models-dir $(DQN_MODELS) \
		--plots-dir plots \
		--prefilter v1

dqn-exp-8: ## Run the 8x8 DQN composition experiment
	$(PYTHON) dqn/full_experiment.py \
		--mode all \
		--grid-size 8 \
		--obs-mode local --local-size 7 \
		--prefilter v1 \
		--base-dir $(DQN_BASE_DIR_8) \
		--minigrid-ids-path $(DQN_RUNS_8)/minigrid_ids.json \
		--eval-seeds-path $(DQN_BASE_DIR_8)/eval_env_seeds.json \
		--faiss-base-dir $(DQN_FAISS) \
		--models-dir $(DQN_MODELS) \
		--data-dir $(DQN_DATA) \
		--results-dir $(DQN_RESULTS)

dqn-repro-8: dqn-train-8 dqn-prep-8 dqn-exp-8 ## Run the full 8x8 DQN workflow

ppo-train-8: ## Train the 8x8 PPO library with the documented experiment settings
	$(PYTHON) policy_reusability/data_generation/deeprl/train_ppo.py \
		--output-root $(PPO_RUNS_8) \
		--grid-preset grid8 \
		--timesteps 2000000 \
		--obs-mode local --local-size 7 \
		--snapshot-interval 0 --snapshot-steps 50000 \
		--n-envs 4 \
		--batch-size 64 \
		--loss-plot \
		--overwrite \
		--all-rewards

ppo-prep-8: ## Build pi2vec assets for the 8x8 PPO library
	$(PYTHON) ppo/pi2vec_preparation.py \
		--base-dir $(PPO_BASE_DIR_8) \
		--minigrid-ids-path $(PPO_RUNS_8)/minigrid_ids.json \
		--index-dir $(PPO_FAISS) \
		--data-dir $(PPO_DATA) \
		--models-dir $(PPO_MODELS) \
		--plots-dir plots \
		--prefilter v2

ppo-exp-8: ## Run the 8x8 PPO composition experiment
	$(PYTHON) ppo/full_experiment.py \
		--mode all \
		--grid-size 8 \
		--obs-mode local --local-size 7 \
		--prefilter v2 \
		--base-dir $(PPO_BASE_DIR_8) \
		--minigrid-ids-path $(PPO_RUNS_8)/minigrid_ids.json \
		--eval-seeds-path $(PPO_BASE_DIR_8)/eval_env_seeds.json \
		--faiss-base-dir $(PPO_FAISS) \
		--models-dir $(PPO_MODELS) \
		--data-dir $(PPO_DATA) \
		--results-dir $(PPO_RESULTS)

ppo-repro-8: ppo-train-8 ppo-prep-8 ppo-exp-8 ## Run the full 8x8 PPO workflow
