#!/bin/bash
set -e

# Capture output to a temporary file
output=$(mktemp)
trap "rm -f $output" EXIT

# Run clean and rebuild in silent mode, capturing all output from both steps.
if { make -s clean && make -s build-dpor-3.14 build-io; } > "$output" 2>&1; then
    echo "Rebuild success"
else
    # If rebuild failed, output the captured error
    cat "$output"
    exit 1
fi
