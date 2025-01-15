#!/bin/bash

# Activate the matlas environment
eval "$(conda shell.bash hook)"
conda activate matlas

# Set number of parallel jobs
N_JOBS=4  # Adjust this number based on available cores

# Create a log directory
mkdir -p logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="logs/search_${timestamp}"
mkdir -p "$log_dir"

# Create a file to track completed combinations
completed_log="${log_dir}/completed_combinations.txt"
touch "$completed_log"

# Generate all parameter combinations with 3 repetitions
{
    for embed_dim in 4 8 16 32; do
        for num_layer in 1 2 3 4; do
            for num_head in 4 8; do
                for dim_ff in 16 32 64; do
                    for num_fourier in 2 4 8 16 32; do
                        for num_epoch in 100; do
                            for lr in 0.1 0.01 0.001 0.0001 0.00001; do
                                for batch_size in 4 8 16 32; do
                                    for rep in {1..3}; do
                                        echo "${embed_dim},${num_layer},${num_head},${dim_ff},${num_fourier},${num_epoch},${lr},${batch_size},${rep}"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
} > "${log_dir}/parameter_combinations.txt"

# Function to run a single experiment
run_experiment() {
    IFS=',' read -r embed_dim num_layer num_head dim_ff num_fourier num_epoch lr batch_size rep <<< "$1"
    local log_dir="$2"  # Accept log_dir as a parameter
    
    log_file="${log_dir}/experiment_${embed_dim}_${num_layer}_${num_head}_${dim_ff}_${num_fourier}_${num_epoch}_${lr}_${batch_size}_rep${rep}.log"
    
    {
        echo "Running with parameters:"
        echo "embed_dim: $embed_dim"
        echo "num_layers: $num_layer"
        echo "num_heads: $num_head"
        echo "dim_feedforward: $dim_ff"
        echo "num_fourier_features: $num_fourier"
        echo "num_epochs: $num_epoch"
        echo "learning_rate: $lr"
        echo "batch_size: $batch_size"
        echo "repetition: $rep"
        
        python train_script.py \
            --embed_dim "$embed_dim" \
            --num_layers "$num_layer" \
            --num_heads "$num_head" \
            --dim_feedforward "$dim_ff" \
            --num_fourier_features "$num_fourier" \
            --num_epochs "$num_epoch" \
            --learning_rate "$lr" \
            --batch_size "$batch_size"
        
        exit_code=$?
        echo "Exit code: $exit_code"
        
        # Log completed combination
        if [ $exit_code -eq 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed: embed_dim=${embed_dim}, num_layer=${num_layer}, num_head=${num_head}, dim_ff=${dim_ff}, num_fourier=${num_fourier}, num_epoch=${num_epoch}, lr=${lr}, batch_size=${batch_size}, rep=${rep}" >> "$completed_log"
        fi
        
        return $exit_code
        
    } &> "$log_file"
}

export -f run_experiment
export completed_log  # Make completed_log available to parallel processes
eval "$(conda shell.bash hook)"
conda activate matlas

# Silence the citation notice
parallel --citation

# Run experiments in parallel, passing log_dir as an additional argument
cat "${log_dir}/parameter_combinations.txt" | parallel -j $N_JOBS "run_experiment {} ${log_dir}"

echo "All experiments completed. Logs are in ${log_dir}"
echo "Check ${completed_log} for a list of all completed combinations"