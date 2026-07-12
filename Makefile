# Telecom Subscriber Segmentation & Churn
#
#   make all      run the full pipeline end to end
#   make clean    remove generated data, models and logs
#   make pages    copy the rendered dashboard into docs/ for GitHub Pages

PYTHON ?= python

.PHONY: all segment predict samples dashboard pages clean distclean help

help:
	@echo "make all       full pipeline (~4 min on one core)"
	@echo "make pages     stage dashboard into docs/ for GitHub Pages"
	@echo "make clean     remove generated artifacts"

all: dashboard

# Stage A: load -> clean -> cluster -> profile -> name. Writes a checkpoint.
segment:
	$(PYTHON) stage_a_segment.py

# Stage B: resumes from the checkpoint. Split from A so each script finishes
# inside a single CPU budget on a one-core machine.
predict: segment
	$(PYTHON) stage_b_predict.py

samples: predict
	$(PYTHON) predict_samples.py

dashboard: samples
	$(PYTHON) build_dashboard_data.py
	@echo ""
	@echo "Open outputs/dashboard.html"

# GitHub Pages serves from docs/ on the default branch.
pages: dashboard
	@mkdir -p docs
	@cp outputs/dashboard.html docs/index.html
	@echo "Staged docs/index.html — commit and enable Pages (source: docs/)."

clean:
	rm -rf data/ models/ *.log
	rm -f outputs/customer_segmentation_output.csv outputs/utg_campaign.csv outputs/ucg_holdout.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

distclean: clean
	rm -rf outputs/ figures/ docs/
