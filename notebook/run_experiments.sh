#!/bin/bash

# Create results directory if it doesn't exist
RESULTS_DIR="experiment_results"
mkdir -p $RESULTS_DIR

# Create a timestamped experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${RESULTS_DIR}/iterations_study_${TIMESTAMP}"
mkdir -p $EXPERIMENT_DIR

# Log file for the entire experiment
LOG_FILE="${EXPERIMENT_DIR}/experiment.log"

# Function to run experiment with specified iterations
run_experiment() {
    local iters=$1
    local strategies=$2
    local output_dir="${EXPERIMENT_DIR}/iter_${iters}"
    mkdir -p $output_dir
    
    echo "Running experiment with ${iters} iterations..." | tee -a $LOG_FILE
    
    # Run both iterative evolution and mixed strategy
    python evaluate.py \
        --strategies $strategies \
        --max-iters $iters \
        --batch-size 4 \
        --output "${output_dir}/results.json" \
        --experiment-name "iteration_study_${iters}" \
        2>&1 | tee -a "${output_dir}/run.log"
    
    echo "Completed experiment with ${iters} iterations" | tee -a $LOG_FILE
}

# Record experiment configuration
echo "Starting iteration parameter study at $(date)" > $LOG_FILE
echo "Results will be saved in: $EXPERIMENT_DIR" | tee -a $LOG_FILE

# Run experiments for different iteration numbers
for iters in 3 4; do
    run_experiment $iters "iterative_evolution mixed"
done

# Combine all results into a single summary JSON
echo "Creating summary of all experiments..." | tee -a $LOG_FILE
python - <<EOF
import json
import glob
import os

def create_summary():
    summary = {
        "experiment_timestamp": "$TIMESTAMP",
        "iteration_results": {}
    }
    
    # Collect results from each iteration
    for iters in [1, 2, 3, 4]:
        result_file = "${EXPERIMENT_DIR}/iter_{}/results.json".format(iters)
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                summary["iteration_results"][str(iters)] = data
    
    # Save summary
    summary_file = "${EXPERIMENT_DIR}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Summary saved to: {}".format(summary_file))

create_summary()
EOF

echo "Experiment complete! Results are in: $EXPERIMENT_DIR" | tee -a $LOG_FILE
echo "Summary of results is in: $EXPERIMENT_DIR/summary.json" | tee -a $LOG_FILE