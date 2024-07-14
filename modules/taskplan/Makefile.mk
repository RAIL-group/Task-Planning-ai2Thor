help::
	@echo "Object Search using LSP-GNN on procthor-10k data(proc-graph):"
	@echo "  gen-data	  Generate graph data from procthor maps."
	@echo "  eval-learned  Evaluates learned planner."
	@echo "  eval-known	  Evaluates known planner."
	@echo "  eval-naive	  Evaluates naive planner."

# --- === Object Search and Task Planning in ProcTHOR === ---#
BASENAME ?= taskplan
NUM_TRAIN_SEEDS ?= 2#2000
NUM_TEST_SEEDS ?= 0#500
NUM_EVAL_SEEDS ?= 1#200

CORE_ARGS ?= --resolution 0.05


### Target for experiments ###
# Data generation target
data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(NUM_TRAIN_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(BASENAME)/data_completion_logs/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 6000 $$((6000 + $(NUM_TEST_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(BASENAME)/data_completion_logs/data_testing_$${ii}.png"; done)

$(data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)

$(data-gen-seeds):
	@echo "Generating Data [$(BASENAME) | seed: $(seed) | $(traintest)"]
	@-rm -f $(DATA_BASE_DIR)/$(BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(BASENAME)/data_completion_logs
	@$(DOCKER_PYTHON) -m taskplan.scripts.gen_data \
		$(CORE_ARGS) \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(BASENAME)

.PHONY: gen-data
gen-data: $(data-gen-seeds)

# Network training target
train-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/gnn.pt
$(train-file): $(data-gen-seeds)
	@$(DOCKER_PYTHON) -m taskplan.scripts.train \
		--num_steps 40000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--save_dir /data/$(BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(BASENAME)/

.PHONY: train
train: $(train-file)
##############################


### Targets for 3rd-party required downloads ###
# Procthor 10k dataset download target
.PHONY: download-procthor-10k
download-procthor-10k:
	@mkdir -p $(DATA_BASE_DIR)/procthor-data/
	@$(DOCKER_PYTHON) -m taskplan.scripts.download \
		--save_dir /data/procthor-data

# Sentesence Bert model download target
.PHONY: download-sbert
download-sbert:
	@mkdir -p $(DATA_BASE_DIR)/sentence_transformers/
	@$(DOCKER_PYTHON) -m taskplan.scripts.download \
		--save_dir /data/sentence_transformers

# Combined download target
.PHONY: download
download:
	$(MAKE) download-procthor-10k
	$(MAKE) download-sbert
################################################