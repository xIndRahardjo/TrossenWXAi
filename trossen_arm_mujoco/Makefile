.PHONY: data train evaluate clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rl_wxai
PYTHON = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
install:
	pip install -e .

## Download / generate raw data
data:
	$(PYTHON) -m $(PROJECT_NAME).dataset

## Train the GadgetArm stage-1 model
train:
	$(PYTHON) wxai_gadgetarm_tinker/train_gadgetarm.py

## Train the stage-2 gripping model
train-stage2:
	$(PYTHON) wxai_gadgetarm_tinker/train_stage2_gripping.py

## Evaluate stage-1 model
evaluate:
	$(PYTHON) wxai_gadgetarm_tinker/evaluate_gadgetarm.py

## Evaluate stage-2 model
evaluate-stage2:
	$(PYTHON) wxai_gadgetarm_tinker/evaluate_stage2_gripping.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
lint:
	ruff check $(PROJECT_NAME)

#################################################################################
# Self-documenting commands                                                     #
#################################################################################

help:
	@echo "Available targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
