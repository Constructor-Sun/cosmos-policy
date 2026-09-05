#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PATH="/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH"

cd "$SCRIPT_DIR/../.."

exec python "$SCRIPT_DIR/run.py" "$@"
