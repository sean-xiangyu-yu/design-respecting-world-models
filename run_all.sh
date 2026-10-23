#!/usr/bin/env bash
# Reproduce all reported numbers, tables, and figures.
# Requires data/df_all_participant.csv (see data/README.md).
set -euo pipefail
cd "$(dirname "$0")/code"
python cluster_month_analysis.py --replicates 500 --workers "${WORKERS:-4}"
python validate_cluster_month_quadrature.py
python shared_sampling_manuscript.py
