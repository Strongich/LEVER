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

.PHONY: help sync lock \
	states-8-0 states-8-99 states-16-0 states-16-99 \
	prep-8-0 prep-8-99 prep-16-0 prep-16-99 \
	exp-8-0 exp-8-99 exp-16-0 exp-16-99 \
	plots-8-0 plots-8-99 plots-16-0 plots-16-99 \
	sweep-8-0 sweep-8-99 sweep-16-0 sweep-16-99 \
	repro-8-0 repro-8-99 repro-16-0 repro-16-99 repro-all

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
