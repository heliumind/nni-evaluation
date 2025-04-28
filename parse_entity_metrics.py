import os
import pandas as pd
import argparse
from collections import defaultdict

def parse_predict_metrics(base_dir, metric, print_results=False):
    # Use a dictionary to organize results by dataset
    dataset_results = defaultdict(list)
    
    # Store all entity metrics we encounter to create consistent headers
    all_entity_metrics = set()

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
                    entity_metrics = {}
                    
                    # Standard metrics to ignore (we're only interested in entity-specific metrics)
                    ignore_metrics = {
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
                        ignore_metrics.update({
                            "predict_overall_accuracy": None,
                        })
                    else:
                        ignore_metrics.update({
                            "predict_accuracy": None,
                        })

                    # Extract entity-specific metrics from the best row
                    for key in best_row.keys():
                        # Look for predict metrics that are not in the ignored_metrics list
                        if key.startswith("predict") and key not in ignore_metrics:
                            # Add the metric to our metrics dictionary
                            entity_metrics[key] = best_row[key]
                            # Record this entity metric for later use in headers
                            all_entity_metrics.add(key)
                    
                    # Add the model's results to the appropriate dataset
                    dataset_results[dataset_name].append({
                        "model_name": model_name,
                        **entity_metrics
                    })
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Ensure the output directory exists
    output_dir = os.path.join(base_dir, "csv")
    os.makedirs(output_dir, exist_ok=True)

    # For each dataset, create a CSV with models as rows and entity metrics as columns
    for dataset_name, models in dataset_results.items():
        if print_results:
            print(f"\nDataset: {dataset_name}")
            for model in models:
                print(f"  Model: {model['model_name']}")
                for metric, value in model.items():
                    if metric != "model_name":
                        print(f"    {metric}: {value}")
        else:
            # Convert to DataFrame and save as CSV
            output_file = os.path.join(output_dir, f"best_{dataset_name}_metrics.csv")
            df = pd.DataFrame(models)
            df.set_index('model_name', inplace=True)
            df.to_csv(output_file)
            print(f"Results for dataset {dataset_name} saved to {output_file}")
    
    print(f"Total datasets: {len(dataset_results)}")
    # print(f"Entity metrics found: {', '.join(sorted(all_entity_metrics))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse predict entity metrics from results.csv files.")
    parser.add_argument("base_dir", type=str, help="Path to directory to search for results.csv files.")
    parser.add_argument("--metric", type=str, default="eval_micro_f1", help="Metric to optimize.")
    parser.add_argument("--print", action="store_true", help="Print the results instead of saving to a CSV file.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    parse_predict_metrics(base_dir, args.metric, print_results=args.print)
