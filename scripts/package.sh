#!/bin/sh
# Build the submission archive: source, configs, docs and tests only.
# Excludes the virtualenv, generated data, caches, build metadata and anything secret-like.
set -eu
cd "$(dirname "$0")/.."
out=dist/orgchart-submission.zip
rm -f "$out"
mkdir -p dist
zip -rq "$out" . \
  -x '.venv/*' 'venv/*' 'data/*' 'dist/*' 'build/*' '.git/*' \
     '.hypothesis/*' '.pytest_cache/*' '.ruff_cache/*' '.mypy_cache/*' '.coverage' \
     '*__pycache__/*' '*.pyc' '*.pyo' '*.egg-info/*' \
     '.DS_Store' '*/.DS_Store' '.env' '.env.*' '.streamlit/secrets.toml' '.idea/*' '.vscode/*'
echo "Wrote $out ($(unzip -l "$out" | tail -1 | awk '{print $2}') files)"
