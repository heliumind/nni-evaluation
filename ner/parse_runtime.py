import os
import re
import pandas as pd

def parse_runtime(base_dir, print_results=False):
    results = []

    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "trial.log":
                file_path = os.path.join(root, file)
                if not os.path.exists(file_path):
                    continue  # Skip if the file does not exist

                try:
                    # Initialize variables
                    train_runtime = None
                    predict_runtime = None

                    # Read the log file
                    with open(file_path, "r") as f:
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
                    if train_runtime and predict_runtime:
                        results.append({
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "train_runtime": train_runtime,
                            "predict_runtime": predict_runtime
                        })
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save results to a CSV file or print them
    if print_results:
        for result in results:
            print(f"Model: {result['model_name']}, Dataset: {result['dataset_name']}, "
              f"Train Runtime: {result['train_runtime']}, Predict Runtime: {result['predict_runtime']}")
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
    import argparse

    parser = argparse.ArgumentParser(description="Parse train and predict runtimes from trial.log files.")
    parser.add_argument("--print", action="store_true", help="Print the results instead of saving to a CSV file.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parse_runtime(base_dir, print_results=args.print)
