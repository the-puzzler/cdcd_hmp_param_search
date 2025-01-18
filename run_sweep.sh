#!/bin/bash

# Number of agents to run
NUM_AGENTS=4

# Name of your Python script
PYTHON_SCRIPT="train_script_wbsweep_save_models.py"

# Loop to start each agent
for ((i=1; i<=NUM_AGENTS; i++))
do
    echo "Starting agent $i"
    python $PYTHON_SCRIPT &
done

# Wait for all background processes to complete
wait

echo "All agents completed"