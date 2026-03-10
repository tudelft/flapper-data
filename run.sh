#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"
PYTHON="python"
FLIGHTS=(hover1 hover2 climb1 climb2 lateral1 lateral2 longitudinal1 longitudinal2 yaw1 yaw2)

# Activate venv if present
if [[ -d "$VENV" ]]; then
    export PATH="$VENV/bin:$PATH"
fi

usage() {
    cat <<EOF
Usage: $0 <command> [flight]

Commands:
  process <flight>    Process a specific flight
  process-all         Process all flights
  rerun <flight>      Rerun visuals for a specific flight
  rerun-all           Rerun visuals for all flights

Available flights: ${FLIGHTS[*]}
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    process)
        flight="${2:-hover1}"
        echo "Processing $flight..."
        $PYTHON process_data.py "$flight"
        ;;
    process-all)
        for flight in "${FLIGHTS[@]}"; do
            echo "Processing $flight..."
            $PYTHON process_data.py "$flight"
        done
        ;;
    rerun)
        flight="${2:-hover1}"
        echo "Rerunning visuals for $flight..."
        $PYTHON rerun_visuals.py "$flight"
        ;;
    rerun-all)
        for flight in "${FLIGHTS[@]}"; do
            echo "Rerunning visuals for $flight..."
            $PYTHON rerun_visuals.py "$flight"
        done
        ;;
    *)
        usage
        ;;
esac
