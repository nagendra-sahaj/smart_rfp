#!/bin/bash
# Shortcut script to run the unified CLI with uv
set -euo pipefail

# Ensure Python can import the 'src' package by adding project root to PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

uv run python -m src.cli.contracts_cli