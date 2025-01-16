#!/bin/bash

# Array to store background process PIDs
declare -a PIDS

# Function to kill all child processes
cleanup() {
    echo "Stopping all agents..."
    # Kill all background processes
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null
    done
    # Wait for processes to terminate
    wait
    echo "All agents stopped"
    exit 1
}

# Set up trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# Parse command line arguments
SWEEP_ID=""
NUM_AGENTS=1
RUNS_PER_AGENT=500

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep_id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --num_agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --runs_per_agent)
            RUNS_PER_AGENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# If no sweep ID provided, create new sweep and capture its ID
if [ -z "$SWEEP_ID" ]; then
    echo "Creating new sweep..."
    # Run Python script with --create_only flag
    OUTPUT=$(python sweep_agent.py --create_only)
    SWEEP_ID=$(echo "$OUTPUT" | grep -o 'SWEEP_ID_START:.*:SWEEP_ID_END' | sed 's/SWEEP_ID_START:\(.*\):SWEEP_ID_END/\1/')
    
    if [ -z "$SWEEP_ID" ]; then
        echo "Failed to create sweep or capture sweep ID"
        exit 1
    fi
    
    echo "Created sweep with ID: $SWEEP_ID"
    
    # Save sweep ID to file
    echo "$SWEEP_ID" > latest_sweep_id.txt
    echo "Sweep ID saved to latest_sweep_id.txt"
fi

# Launch the specified number of agents
for ((i=1; i<=$NUM_AGENTS; i++)); do
    echo "Starting agent $i for sweep $SWEEP_ID..."
    python sweep_agent.py --sweep_id "$SWEEP_ID" --count "$RUNS_PER_AGENT" &
done

# Wait for all background processes to complete
wait

echo "All agents completed"