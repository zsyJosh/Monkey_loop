#!/bin/bash

# Create results directory if it doesn't exist
RESULTS_DIR="experiment_results"
mkdir -p $RESULTS_DIR

# Create a timestamped experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${RESULTS_DIR}/batch_size_study_${TIMESTAMP}"
mkdir -p $EXPERIMENT_DIR

# Log file for the entire experiment
LOG_FILE="${EXPERIMENT_DIR}/experiment.log"

# Function to run experiment with specified batch size
run_experiment() {
    local batch_size=$1
    local strategies=$2
    local output_dir="${EXPERIMENT_DIR}/batch_${batch_size}"
    mkdir -p $output_dir
    
    echo "Running experiment with batch size ${batch_size}..." | tee -a $LOG_FILE
    
    # Run both batch sampling and mixed strategy
    python evaluate.py \
        --strategies $strategies \
        --batch-size $batch_size \
        --max-iters 2 \
        --output "${output_dir}/results.json" \
        --experiment-name "batch_study_${batch_size}" \
        2>&1 | tee -a "${output_dir}/run.log"
    
    echo "Completed experiment with batch size ${batch_size}" | tee -a $LOG_FILE
}

# Record experiment configuration
echo "Starting batch size parameter study at $(date)" > $LOG_FILE
echo "Results will be saved in: $EXPERIMENT_DIR" | tee -a $LOG_FILE

# Run experiments for different batch sizes
for batch_size in 8; do
    run_experiment $batch_size "batch_sampling mixed"
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
        "batch_size_results": {}
    }
    
    # Collect results from each batch size
    for batch_size in [2, 4, 6, 8, 10]:
        result_file = "${EXPERIMENT_DIR}/batch_{}/results.json".format(batch_size)
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                summary["batch_size_results"][str(batch_size)] = data
    
    # Save summary
    summary_file = "${EXPERIMENT_DIR}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create a CSV summary for easy analysis
    csv_file = "${EXPERIMENT_DIR}/summary.csv"
    with open(csv_file, 'w') as f:
        # Write header
        f.write("batch_size,strategy,performance,execution_time,cost\n")
        
        # Write data for each batch size and strategy
        for batch_size, results in summary["batch_size_results"].items():
            for strategy, metrics in results["strategies"].items():
                if metrics["status"] == "success":
                    f.write(f"{batch_size},{strategy}," + 
                           f"{metrics.get('performance', 'N/A')}," +
                           f"{metrics.get('execution_time', 'N/A')}," +
                           f"{metrics.get('cost', 'N/A')}\n")
    
    print("Summary saved to: {}".format(summary_file))
    print("CSV summary saved to: {}".format(csv_file))

create_summary()
EOF

echo "Experiment complete! Results are in: $EXPERIMENT_DIR" | tee -a $LOG_FILE
echo "Summary of results is in: $EXPERIMENT_DIR/summary.json" | tee -a $LOG_FILE
echo "CSV summary is in: $EXPERIMENT_DIR/summary.csv" | tee -a $LOG_FILE

# Create a quick performance plot using Python and matplotlib
echo "Creating performance visualization..." | tee -a $LOG_FILE
python - <<EOF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV summary
df = pd.read_csv("${EXPERIMENT_DIR}/summary.csv")

# Create performance plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='batch_size', y='performance', hue='strategy', marker='o')
plt.title('Performance vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Performance Score')
plt.grid(True)
plt.savefig("${EXPERIMENT_DIR}/performance_plot.png")

# Create cost plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='batch_size', y='cost', hue='strategy', marker='o')
plt.title('Cost vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Cost ($)')
plt.grid(True)
plt.savefig("${EXPERIMENT_DIR}/cost_plot.png")

# Create execution time plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='batch_size', y='execution_time', hue='strategy', marker='o')
plt.title('Execution Time vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Execution Time (s)')
plt.grid(True)
plt.savefig("${EXPERIMENT_DIR}/time_plot.png")
EOF

echo "Visualizations have been created in: $EXPERIMENT_DIR" | tee -a $LOG_FILE