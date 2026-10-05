#!/usr/bin/env bash
set -euo pipefail

PYTHON="uv run python -m"
FLIGHTS=(flight_001 flight_002 hover1 hover2 climb1 climb2 lateral1 lateral2 longitudinal1 longitudinal2 yaw1 yaw2)

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
        $PYTHON flapper_data.process_data "$flight"
        ;;
    process-all)
        for flight in "${FLIGHTS[@]}"; do
            echo "Processing $flight..."
            $PYTHON flapper_data.process_data "$flight"
        done
        ;;
    rerun)
        flight="${2:-hover1}"
        echo "Rerunning visuals for $flight..."
        $PYTHON flapper_data.rerun_visuals "$flight"
        ;;
    rerun-all)
        for flight in "${FLIGHTS[@]}"; do
            echo "Rerunning visuals for $flight..."
            $PYTHON flapper_data.rerun_visuals "$flight"
        done
        ;;
    *)
        usage
        ;;
esac
