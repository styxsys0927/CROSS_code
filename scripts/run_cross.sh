#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
SEEDS="${SEEDS:-0}"
DATASETS="${DATASETS:-BCIC-IV-2a BCIC-IV-2b Zhou2016 OpenBMI PhysioNet-MI Mumtaz2017 ADFTD Rockhill2021 EEGMat Shin2018}"
DRY_RUN=0
STOP_ON_ERROR=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cross.sh [options]

Options:
  --python PATH          Python executable (default: python)
  --device ID            CUDA device id (default: 0)
  --dataset LIST         Dataset list, comma- or space-separated
  --datasets LIST        Same as --dataset
  --seed LIST            Seed list, comma- or space-separated
  --seeds LIST           Same as --seed
  --dry-run              Print commands without running
  --stop-on-error        Stop immediately if one run fails
  -h, --help             Show this help
EOF
}

normalize_list() {
  echo "$1" | tr ',' ' '
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dataset|--datasets) DATASETS="$(normalize_list "$2")"; shift 2 ;;
    --seed|--seeds) SEEDS="$(normalize_list "$2")"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --stop-on-error) STOP_ON_ERROR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

trainer_for_dataset() {
  case "$1" in
    BCIC-IV-2a|BCIC-IV-2b|Zhou2016) echo "train_loso.py" ;;
    *) echo "train_kfold.py" ;;
  esac
}

run_command() {
  echo "[CROSS][$1][seed=$2] $PYTHON_BIN $3 --config config/cross.yaml --dataset $1 --seed $2 --device $DEVICE"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if "$PYTHON_BIN" "$3" --config "config/cross.yaml" --dataset "$1" --seed "$2" --device "$DEVICE"; then
    return 0
  fi
  status=$?
  echo "Failed: model=CROSS dataset=$1 seed=$2 exit=$status" >&2
  if [[ "$STOP_ON_ERROR" == "1" ]]; then
    exit "$status"
  fi
}

for dataset in $DATASETS; do
  trainer="$(trainer_for_dataset "$dataset")"
  for seed in $SEEDS; do
    run_command "$dataset" "$seed" "$trainer"
  done
done
