PYTHON ?= python3   # override with a 3.12-3.15 interpreter, e.g. make setup PYTHON=python3.13
PY := .venv/bin/python
ORGCHART := .venv/bin/orgchart

.PHONY: setup test test-fast sample full validate pipeline normalize fit infer evaluate app verify package clean

setup:            ## create venv and install pinned dependencies + package
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements.lock
	.venv/bin/pip install -e '.[dev]' --no-deps

test:             ## full test suite (includes end-to-end generation)
	$(PY) -m pytest

test-fast:        ## unit/property tests only
	$(PY) -m pytest -m "not slow"

sample:           ## small dataset for quick inspection
	$(ORGCHART) generate --out data/sample --n-companies 24 --name sample --force

full:             ## assessment dataset: 600 companies, >=100 observed employees each
	$(ORGCHART) generate --out data/full --force

validate:         ## dataset structural checks + realism report
	$(ORGCHART) validate data/full

pipeline:         ## normalise -> fit -> infer -> check -> evaluate on data/full (add GENERATE=1 to rebuild data)
	$(ORGCHART) pipeline data/full $(if $(GENERATE),--generate,)

normalize:
	$(ORGCHART) normalize data/full

fit:              ## learned weights (train), calibration + thresholds + selection (calibration companies)
	$(ORGCHART) fit data/full

infer:            ## predictions for every observed profile (observed data + model files only)
	$(ORGCHART) infer data/full

evaluate:         ## metrics vs ground truth -> data/full/reports/inference_eval.md
	$(ORGCHART) evaluate data/full

app:              ## interactive dashboard
	.venv/bin/streamlit run app/streamlit_app.py

verify:           ## everything a reviewer needs to trust the build; non-zero exit on any failure
	$(PY) -m pytest
	$(ORGCHART) validate data/full
	$(ORGCHART) check-predictions data/full
	$(ORGCHART) check-predictions data/full --variant naive
	$(ORGCHART) verify data/full

package:          ## submission zip without .venv, data, caches, build metadata or secrets -> dist/
	sh scripts/package.sh

clean:
	rm -rf data/sample data/full .pytest_cache .hypothesis
