import os
import re
import pandas as pd
import argparse
from datetime import timedelta

def parse_runtime(base_dir, metric, nni_dir, print_results=False):
    results = []

    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            experiment_id = None
            if file == "experiment_id.txt":
                file_path = os.path.join(root, file)
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist. Skipping...")
                    continue

                with open(file_path, "r") as f:
                    experiment_id = f.readline().strip()

                # Walk through the trial directories
                trial_dir = os.path.join(nni_dir, experiment_id, "environments", "local-env", "trials")
                if not os.path.exists(trial_dir):
                    continue

                total_runtime = 0
                for trial in os.listdir(trial_dir):
                    trial_log_path = os.path.join(trial_dir, trial, "trial.log")
                    if not os.path.exists(trial_log_path):
                        continue

                    try:
                        with open(trial_log_path, "r") as trial_log:
                            for line in trial_log:
                                if "train_runtime" in line:
                                    match = re.search(r"train_runtime\s+=\s+([\d:.]+)", line)
                                    if match:
                                        # Parse the runtime using datetime.timedelta
                                        time_parts = match.group(1).split(':')
                                        # Format is H:MM:SS.SS
                                        runtime = timedelta(hours=int(time_parts[0]), 
                                                            minutes=int(time_parts[1]), 
                                                            seconds=float(time_parts[2])).total_seconds()
                                        total_runtime += runtime
                    except Exception as e:
                        print(f"Error reading trial log {trial_log_path}: {e}")
                        continue

                file_path = os.path.join(root, "results.csv")
                if not os.path.exists(file_path):
                    continue  # Skip if the file does not exist

                try:
                    df = pd.read_csv(file_path)
                    best_row = df.loc[df[metric].idxmax()]
                    trial_id = best_row['trialJobId']
                    trial_log_path = os.path.join(nni_dir, experiment_id, "environments", "local-env", "trials", trial_id, "trial.log")

                    # Initialize variables
                    train_runtime = None
                    predict_runtime = None

                    # Read the log file
                    with open(trial_log_path, "r") as f:
                        for line in f:
                            # Extract train_runtime
                            if "train_runtime" in line:
                                match = re.search(r"train_runtime\s+=\s+([\d:.]+)", line)
                                if match:
                                    train_runtime = match.group(1)

                            # Extract predict_runtime
                            if "predict_runtime" in line:
                                match = re.search(r"predict_runtime\s+=\s+([\d:.]+)", line)
                                if match:
                                    predict_runtime = match.group(1)

                    # Extract model and dataset names
                    model_name = os.path.basename(os.path.dirname(root))
                    dataset_name = os.path.basename(root)

                    # Append to results if both runtimes are found
                    if train_runtime and predict_runtime and total_runtime:
                        results.append({
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "train_runtime": train_runtime,
                            "predict_runtime": predict_runtime,
                            "total_runtime": str(timedelta(seconds=total_runtime))
                        })
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save results to a CSV file or print them
    if print_results:
        for result in results:
            print(f"Model: {result['model_name']}, Dataset: {result['dataset_name']}, "
              f"Train Runtime: {result['train_runtime']}, Predict Runtime: {result['predict_runtime']}, "
              f"Total Runtime: {result['total_runtime']}")
        print(f"Total experiments: {len(results)}")
    else:
        # Ensure the 'csv' directory exists
        output_dir = os.path.join(base_dir, "csv")
        os.makedirs(output_dir, exist_ok=True)

        # Save the results to the 'csv' directory
        output_file = os.path.join(output_dir, "best_runtimes.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse predict metrics from trial.log files.")
    parser.add_argument("base_dir", type=str, help="Path to directory to search for results.csv files.")
    parser.add_argument("--nni_dir", type=str, default="~/nni-experiments", help="Optional path to NNI directory.")
    parser.add_argument("--metric", type=str, default="predict_micro_f1", help="Metric to optimize.")
    parser.add_argument("--print", action="store_true", help="Print the results instead of saving to a CSV file.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    nni_dir = os.path.abspath(os.path.expanduser(args.nni_dir))
    parse_runtime(base_dir, args.metric, nni_dir=nni_dir, print_results=args.print)
