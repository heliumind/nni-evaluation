import os
import pandas as pd
import argparse

def parse_best_hyperparams(base_dir, print_results=False):
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
                    best_row = df.loc[df['reward'].idxmax()]

                    # Extract relevant information
                    model_name = os.path.basename(os.path.dirname(root))
                    dataset_name = os.path.basename(root)
                    learning_rate = best_row['learning_rate']
                    batch_size = best_row['per_device_train_batch_size']

                    # Append to results
                    results.append({
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size
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
        output_file = os.path.join(output_dir, "best_hyperparams.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse best hyperparameters from results.csv files.")
    parser.add_argument("--print", action="store_true", help="Print the results instead of saving to a CSV file.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parse_best_hyperparams(base_dir, print_results=args.print)
