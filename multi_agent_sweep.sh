#!/bin/bash

SWEEP_ID_FILE="current_sweep.txt"

# Create new sweep if needed
if [ ! -s "$SWEEP_ID_FILE" ]; then
    echo "No existing sweep ID found. Creating new sweep..."
    if ! python train_script_wbsweep.py | grep -i "sweep with ID:" | cut -d ":" -f 2 | tr -d ' ' > "$SWEEP_ID_FILE"; then
        echo "Failed to create new sweep"
        exit 1
    fi
fi

# Verify sweep ID exists
if [ ! -s "$SWEEP_ID_FILE" ]; then
    echo "Failed to get sweep ID"
    exit 1
fi

SWEEP_ID=$(cat "$SWEEP_ID_FILE")
echo "Using sweep ID: $SWEEP_ID"

N_AGENTS=3

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "conda not found"
    exit 1
fi

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
if ! conda activate matlas; then
    echo "Failed to activate conda environment"
    exit 1
fi

# Launch agents
for i in $(seq 1 $N_AGENTS)
do
    echo "Starting agent $i for sweep $SWEEP_ID"
    wandb agent $SWEEP_ID &
done

wait