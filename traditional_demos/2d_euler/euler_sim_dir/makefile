##########
# EDIT BELOW

BASEDIR=/Users/nickm/thesis/InvariantPreservingMLSolvers/traditional_demos/2d_euler/euler_sim_dir

# EDIT ABOVE 
##########


READWRITE_DIR=$(BASEDIR)
SCRIPT_DIR=$(BASEDIR)/code/scripts
POISSON_DIR=$(BASEDIR)
MAKE_SCRIPT=$(SCRIPT_DIR)/make.mk
PYTHONPATH=$(BASEDIR)/code/simcode:$(BASEDIR)/code/analysiscode:$(BASEDIR)/code/scripts:$(BASEDIR)


EULER_ARGS_FILE=$(BASEDIR)/data/args/2d_euler_args.txt
EQ=EULER
UNIQUE_ID = $(EQ)_test
ARGS_FILE = $($(EQ)_ARGS_FILE)

export SCRIPT_DIR READWRITE_DIR POISSON_DIR UNIQUE_ID ARGS_FILE PYTHONPATH

generate_data :
	make -f $(MAKE_SCRIPT) generate_data

plot_data :
	make -f $(MAKE_SCRIPT) plot_data

plot_diagnostics :
	make -f $(MAKE_SCRIPT) plot_diagnostics