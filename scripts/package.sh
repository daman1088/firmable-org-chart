#!/bin/sh
# Build the submission archive: source, configs, docs and tests only.
# Excludes the virtualenv, generated data, caches, build metadata, nested archives
# (e.g. demo-data.zip, the GitHub Release asset) and anything secret-like.
set -eu
cd "$(dirname "$0")/.."
out=dist/orgchart-submission.zip
rm -f "$out"
mkdir -p dist
zip -rq "$out" . \
  -x '.venv/*' 'venv/*' 'data/*' 'dist/*' 'build/*' '.git/*' \
     '.hypothesis/*' '.pytest_cache/*' '.ruff_cache/*' '.mypy_cache/*' '.coverage' \
     '*__pycache__/*' '*.pyc' '*.pyo' '*.egg-info/*' \
     '*.zip' '*.tar' '*.tar.gz' '*.tgz' '*.gz' '*.bz2' '*.xz' '*.7z' '*.whl' \
     '*.parquet' '*.duckdb' '*.sqlite' '*.db' \
     '.DS_Store' '*/.DS_Store' '.env' '.env.*' '.streamlit/secrets.toml' '.idea/*' '.vscode/*'

# Fail loudly if an archive or dataset slipped through.
unzip -tq "$out" >/dev/null
leaked=$(unzip -Z1 "$out" | grep -iE '(\.(zip|tar|tgz|gz|bz2|xz|7z|whl|parquet|duckdb|sqlite|db)$)|(^|/)data/' || true)
if [ -n "$leaked" ]; then
  echo "error: $out contains archives or generated data:" >&2
  echo "$leaked" >&2
  exit 1
fi
echo "Wrote $out ($(unzip -l "$out" | tail -1 | awk '{print $2}') files)"
