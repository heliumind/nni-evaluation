import os
import pandas as pd
import argparse

def parse_predict_metrics(base_dir, metric, print_results=False):
    results = []

    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "results.csv":
                file_path = os.path.join(root, file)
                if not os.path.exists(file_path):
                    continue  # Skip if the file does not exist

                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Find the row with the highest reward
                    best_row = df.loc[df[metric].idxmax()]

                    # Extract relevant information
                    model_name = os.path.basename(os.path.dirname(root))
                    dataset_name = os.path.basename(root)
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
                    }
                    # Adjust accuracy metric name based on the task
                    if "ner" in base_dir:
                        metrics.update({
                            "predict_overall_accuracy": None,
                        })
                    else:
                        metrics.update({
                            "predict_accuracy": None,
                        })

                    # Extract metrics from the best row
                    for key in metrics.keys():
                        if key in best_row:
                            metrics[key] = best_row[key]

                    # Append to results
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
                  f"Learning Rate: {result['learning_rate']}, Batch Size: {result['batch_size']}")
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
    parser = argparse.ArgumentParser(description="Parse predict metrics from results.csv files.")
    parser.add_argument("base_dir", type=str, help="Path to directory to search for results.csv files.")
    parser.add_argument("--metric", type=str, default="predict_micro_f1", help="Metric to optimize.")
    parser.add_argument("--print", action="store_true", help="Print the results instead of saving to a CSV file.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    parse_predict_metrics(base_dir, args.metric, print_results=args.print)
