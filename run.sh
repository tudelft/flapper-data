#!/usr/bin/env bash
set -euo pipefail

PYTHON="uv run python -m"

usage() {
    cat <<EOF
Usage: $0 <command> [flight]

Commands:
  process <flight>    Process a specific flight
  process-all         Process all flights
  rerun <flight>      Rerun visuals for a specific flight
  rerun-all           Rerun visuals for all flights

Add --processed to the rerun commands to show the processed file instead of the raw mocap.

Available flights: $($PYTHON flapper_data.data_loader)
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    process)
        [[ $# -lt 2 ]] && usage
        $PYTHON flapper_data.process_data "$2"
        ;;
    process-all)
        $PYTHON flapper_data.process_data
        ;;
    rerun)
        [[ $# -lt 2 ]] && usage
        $PYTHON flapper_data.rerun_visuals "${@:2}"
        ;;
    rerun-all)
        for flight in $($PYTHON flapper_data.data_loader); do
            echo "Rerunning visuals for $flight..."
            $PYTHON flapper_data.rerun_visuals "$flight" "${@:2}"
        done
        ;;
    *)
        usage
        ;;
esac
