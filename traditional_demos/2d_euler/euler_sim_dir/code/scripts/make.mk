# DIRECTORIES
EVAL_DATA_DIR = $(READWRITE_DIR)/data/evaldata
EVAL_DIR = $(EVAL_DATA_DIR)/$(UNIQUE_ID)


# SCRIPTS
EVAL_DATA_SCRIPT = $(SCRIPT_DIR)/generate_data.py
PLOT_SCRIPT = $(SCRIPT_DIR)/plot_data.py
DIAGNOSTICS_SCRIPT = $(SCRIPT_DIR)/plot_diagnostics.py

ARGS = --eval_dir $(EVAL_DIR) --poisson_dir $(POISSON_DIR) -id $(UNIQUE_ID) $(shell cat $(ARGS_FILE)) 

generate_data :
	-mkdir $(EVAL_DIR)
	-cp $(ARGS_FILE) $(EVAL_DIR)
	python $(EVAL_DATA_SCRIPT) $(ARGS)

plot_data :
	python $(PLOT_SCRIPT) $(ARGS)

plot_diagnostics :
	python $(DIAGNOSTICS_SCRIPT) $(ARGS)
