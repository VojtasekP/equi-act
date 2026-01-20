#!/usr/bin/env bash
set -euo pipefail

DEFAULT_YAML="src/swesps/sweep_colorectal.yaml"

usage() {
  cat <<'EOF'
Usage:
  ./run_sweep_multi.sh [SWEEP_YAML] --cuda0 N --cuda1 M [--cudaX K ...]
  ./run_sweep_multi.sh --sweep path/to/sweep.yaml --cuda0 N --cuda1 M [--cudaX K ...]

Examples:
  ./run_sweep_multi.sh src/SO2_Nets/sweep_configs/sweep_colorectal.yaml --cuda0 8 --cuda1 8
  ./run_sweep_multi.sh --sweep src/SO2_Nets/sweep_configs/sweep_mnist_final.yaml --cuda0 4
  ./run_sweep_multi.sh --cuda0 4    # uses default YAML: src/SO2_Nets/sweep_configs/sweep_colorectal.yaml

Notes:
  - Runs `wandb sweep` once, extracts "<entity>/<project>/<sweep_id>",
    then launches N agents on each specified GPU with CUDA_VISIBLE_DEVICES.
  - Ctrl-C will terminate all child agents.
EOF
}

YAML="${DEFAULT_YAML}"
YAML_SET=0

declare -A GPU_COUNTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)
      [[ -z "${2:-}" || "${2}" == --* ]] && { echo "Missing YAML path after --sweep"; exit 2; }
      YAML="$2"
      YAML_SET=1
      shift 2
      ;;
    --sweep=*)
      YAML="${1#--sweep=}"
      [[ -z "$YAML" ]] && { echo "Missing YAML path after --sweep="; exit 2; }
      YAML_SET=1
      shift
      ;;
    --cuda[0-9]*)
      gpu="${1#--cuda}"
      [[ -z "${2:-}" ]] && { echo "Missing count after $1"; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "Count for $1 must be an integer"; exit 2; }
      GPU_COUNTS["$gpu"]="$2"
      shift 2
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      if [[ "$1" != --* && "$YAML_SET" -eq 0 ]]; then
        YAML="$1"
        YAML_SET=1
        shift
        continue
      fi
      echo "Unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ ${#GPU_COUNTS[@]} -eq 0 ]]; then
  echo "No GPUs specified. Give flags like --cuda0 8 --cuda1 8"
  usage
  exit 2
fi

if [[ ! -f "$YAML" ]]; then
  echo "YAML not found: $YAML"
  exit 2
fi

command -v wandb >/dev/null 2>&1 || { echo "[error] wandb CLI not found in PATH"; exit 2; }

echo "[info] Creating sweep from: $YAML"
SWEEP_OUTPUT="$(wandb sweep "$YAML" 2>&1 || true)"
echo "$SWEEP_OUTPUT"

# Robust extraction
AGENT_SPEC="$(
  echo "$SWEEP_OUTPUT" \
  | sed -n 's/.*wandb[[:space:]]\+agent[[:space:]]\+\([^[:space:]]\+\).*/\1/p' \
  | tail -n1
)"
if [[ -z "$AGENT_SPEC" ]]; then
  AGENT_SPEC="$(
    echo "$SWEEP_OUTPUT" \
    | sed -n 's/.*Created[^:]*:[[:space:]]\+\([^[:space:]]\+\).*/\1/p' \
    | tail -n1
  )"
fi
if [[ -z "$AGENT_SPEC" ]]; then
  echo "[error] Failed to extract agent spec from wandb sweep output."
  exit 1
fi

echo "[info] Using agent: $AGENT_SPEC"

PIDS=()
cleanup() {
  echo
  echo "[info] Cleaning up ${#PIDS[@]} agents..."
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM

for gpu in "${!GPU_COUNTS[@]}"; do
  count="${GPU_COUNTS[$gpu]}"
  if [[ "$count" -le 0 ]]; then
    echo "[warn] Skipping GPU $gpu with non-positive count: $count"
    continue
  fi
  for ((i=1; i<=count; i++)); do
    echo "[info] Launching: CUDA_VISIBLE_DEVICES=$gpu wandb agent $AGENT_SPEC"
    CUDA_VISIBLE_DEVICES="$gpu" wandb agent "$AGENT_SPEC" &
    PIDS+=("$!")
    sleep 0.25
  done
done

echo "[info] Launched ${#PIDS[@]} agents. Waiting..."
wait
echo "[info] All agents finished."
