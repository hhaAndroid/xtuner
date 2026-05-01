#!/usr/bin/env bash
# Usage:
#   ./run.sh                      # run all tasks
#   ./run.sh offer-letter-generator                  # run one task
#   ./run.sh task-a,task-b,task-c                    # run specific tasks (comma-separated)
#   ./run.sh --n 5                # run first N tasks alphabetically
#   ./run.sh --n 5 --shuffle      # run N tasks picked randomly

set -euo pipefail

TASKS_DIR="${TASKS_DIR:-/mnt/shared-storage-user/huanghaian/code/bench/skillsbench/tasks}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/mnt/shared-storage-user/huanghaian/code/xtuner/worker_logs/skillsbench_artifacts_36_1}"

gateway_url="${GATEWAY_URL:-http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1}"
api_key="${API_KEY:-sk-admin}"
model_name="${MODEL_NAME:-xtuner_gateway_demo}"
max_concurrent="${MAX_CONCURRENT:-32}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
task_names=""
n_tasks=""
shuffle=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)      n_tasks="$2"; shift 2 ;;
        --shuffle) shuffle=true; shift ;;
        --*)      echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            # Positional: explicit task name(s), comma-separated
            task_names="$1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve task list
# ---------------------------------------------------------------------------
if [[ -z "$task_names" ]]; then
    # Collect all valid task names from TASKS_DIR
    all_tasks=$(
        for d in "$TASKS_DIR"/*/; do
            name=$(basename "$d")
            [[ -f "$d/task.toml" && -f "$d/instruction.md" && -f "$d/tests/test.sh" ]] && echo "$name"
        done | sort
    )

    if [[ -n "$n_tasks" ]]; then
        if $shuffle; then
            task_names=$(echo "$all_tasks" | shuf | head -n "$n_tasks" | paste -sd,)
        else
            task_names=$(echo "$all_tasks" | head -n "$n_tasks" | paste -sd,)
        fi
    fi
    # If still empty, run all (pass empty string = all)
fi

# ---------------------------------------------------------------------------
# Derive output paths
# ---------------------------------------------------------------------------
if [[ -z "$task_names" ]]; then
    run_label="all"
else
    # Use first task name as label if multiple, to keep paths short
    first=$(echo "$task_names" | cut -d, -f1)
    count=$(echo "$task_names" | tr ',' '\n' | wc -l)
    run_label=$([ "$count" -eq 1 ] && echo "$first" || echo "${first}_and_${count}")
fi

timestamp=$(date +%Y%m%d_%H%M%S)
work_dir="${ARTIFACTS_DIR}/${run_label}_${timestamp}"
output_file="${work_dir}/results.json"

mkdir -p "$work_dir"

echo "Tasks      : ${task_names:-all (${ARTIFACTS_DIR})}"
echo "Work dir   : ${work_dir}"
echo "Concurrent : ${max_concurrent}"
echo ""

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python recipe/claude_code_sandbox_demo/skillsbench_eval.py \
    --gateway-url "${gateway_url}" \
    --api-key "${api_key}" \
    --model-name "${model_name}" \
    --tasks-dir "${TASKS_DIR}" \
    ${task_names:+--task-names "${task_names}"} \
    --max-concurrent "${max_concurrent}" \
    --output-file "${output_file}" \
    --work-dir "${work_dir}"
