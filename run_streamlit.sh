#!/bin/bash
# Shortcut script to run Streamlit app with uv
set -euo pipefail

# Ensure Python can import the 'src' package by adding project root to PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

uv run streamlit run src/ui/app.py