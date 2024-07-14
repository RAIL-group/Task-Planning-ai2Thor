help::
	@echo "Object Search using LSP-GNN on procthor-10k data(proc-graph):"
	@echo "  proc-graph-gen-data	  Generate graph data from procthor maps."
	@echo "  proc-graph-eval-learned  Evaluates learned planner."
	@echo "  proc-graph-eval-known	  Evaluates known planner."
	@echo "  proc-graph-eval-naive	  Evaluates naive planner."

# --- === Object Search and Task Planning in ProcTHOR === ---#
BASENAME ?= taskplan
NUM_TRAIN_SEEDS ?= 2000
NUM_TEST_SEEDS ?= 500
NUM_EVAL_SEEDS ?= 200


### Target for experiments ###
# Data generation target

##############################
data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(NUM_TRAIN_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(BASENAME)/data_completion_logs/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 6000 $$((6000 + $(NUM_TEST_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(BASENAME)/data_completion_logs/data_testing_$${ii}.png"; done)


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