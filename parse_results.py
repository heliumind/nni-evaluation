import os
import re
import pandas as pd
import argparse
from datetime import timedelta

def parse_results(base_dir, nni_dir):

    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
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

                results = []
                # Extract model and dataset names
                model_name = os.path.basename(os.path.dirname(root))
                dataset_name = os.path.basename(root)
                
                for trial in os.listdir(trial_dir):
                    trial_log_path = os.path.join(trial_dir, trial, "trial.log")
                    if not os.path.exists(trial_log_path):
                        print(f"Trial log {trial_log_path} does not exist. Skipping...")
                        continue
                    
                    # Initialize variables
                    learning_rate = None
                    batch_size = None
                    metrics = {}
                    for mode in ["eval", "predict"]:
                        metrics.update({ 
                            f"{mode}_macro_f1": None,
                            f"{mode}_macro_precision": None,
                            f"{mode}_macro_recall": None,
                            f"{mode}_micro_f1": None,
                            f"{mode}_micro_precision": None,
                            f"{mode}_micro_recall": None,
                            f"{mode}_weighted_f1": None,
                            f"{mode}_weighted_precision": None,
                            f"{mode}_weighted_recall": None,
                        })
                        # Adjust accuracy metric name based on the task
                        if "ner" in base_dir:
                            metrics.update({
                                f"{mode}_overall_accuracy": None,
                            })
                        else:
                            metrics.update({
                                f"{mode}_accuracy": None,
                            })

                    try:
                        with open(trial_log_path, "r") as trial_log:
                            for line in trial_log:
                                if line.startswith("learning_rate"):
                                    learning_rate = float(re.search(r"learning_rate=([\d.eE+-]+)", line).group(1))
                                if line.startswith("per_device_train_batch_size"):
                                    batch_size = int(re.search(r"per_device_train_batch_size=([\d]+)", line).group(1))
                                for metric in metrics.keys():
                                    if metric in line:
                                        match = re.search(rf"{metric}\s+=\s+([\d.]+)", line)
                                        if match:
                                            metrics[metric] = float(match.group(1))

                    except Exception as e:
                        print(f"Error reading trial log {trial_log_path}: {e}")
                        continue
                
                    trial_id = os.path.basename(trial)

                    results.append({
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "trialJobId": trial_id,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        **metrics
                    })

                # Save results to a CSV file
                output_file = os.path.join(base_dir, "experiments", model_name, dataset_name, "results.csv")
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse predict metrics from trial.log files.")
    parser.add_argument("base_dir", type=str, help="Path to directory to search for experiment ids.")
    parser.add_argument("--nni_dir", type=str, default="~/nni-experiments", help="Optional path to NNI directory.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    nni_dir = os.path.abspath(os.path.expanduser(args.nni_dir))
    parse_results(base_dir, nni_dir=nni_dir)
