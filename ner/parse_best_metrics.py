import os
import re
import pandas as pd

def parse_predict_metrics(base_dir, print_results=False):
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
                    metrics = {
                        "predict_macro_f1": None,
                        "predict_macro_precision": None,
                        "predict_macro_recall": None,
                        "predict_micro_f1": None,
                        "predict_micro_precision": None,
                        "predict_micro_recall": None,
                        "predict_weighted_f1": None,
                        "predict_weighted_precision": None,
                        "predict_weighted_recall": None,
                        "predict_overall_accuracy": None
                    }

                    # Read the log file
                    with open(file_path, "r") as f:
                        for line in f:
                            for metric in metrics.keys():
                                if metric in line:
                                    match = re.search(rf"{metric}\s+=\s+([\d.]+)", line)
                                    if match:
                                        metrics[metric] = float(match.group(1))

                    # Extract model and dataset names
                    model_name = os.path.basename(os.path.dirname(root))
                    dataset_name = os.path.basename(root)

                    # Append to results if all metrics are found
                    if all(value is not None for value in metrics.values()):
                        results.append({
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            **metrics
                        })
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save results to a CSV file or print them
    if print_results:
        for result in results:
            print(f"Model: {result['model_name']}, Dataset: {result['dataset_name']}, "
                  f"Metrics: {', '.join([f'{k}: {v}' for k, v in result.items() if k not in ['model_name', 'dataset_name']])}")
        print(f"Total experiments: {len(results)}")
    else:
        # Ensure the 'csv' directory exists
        output_dir = os.path.join(base_dir, "csv")
        os.makedirs(output_dir, exist_ok=True)

        # Save the results to the 'csv' directory
        output_file = os.path.join(output_dir, "best_metrics.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse predict metrics from trial.log files.")
    parser.add_argument("--print", action="store_true", help="Print the results instead of saving to a CSV file.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parse_predict_metrics(base_dir, print_results=args.print)
