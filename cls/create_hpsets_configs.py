import os
import json

# Available GPU devices
CUDA_VISIBLE_DEVICES = 2

models = {
    "christbert_best": "/home/data/models/ChristBERT/hf/model/best",
    "christbert_last": "/home/data/models/ChristBERT/hf/model/last",
    "christbert_scratch_best": "/home/data/models/ChristBERT/hf/scratch/best",
    "christbert_scratch_last": "/home/data/models/ChristBERT/hf/scratch/last",
    "christbert_scratch_bpe_best": "/home/data/models/ChristBERT/hf/scratch_bpe/best",
    "christbert_scratch_bpe_last": "/home/data/models/ChristBERT/hf/scratch_bpe/last",
    "medbertde": "GerMedBERT/medbert-512",
    "biogottbert": "SCAI-BIO/bio-gottbert-base",
    "geistbert": "/home/data/models/GeistBERT",
    "geberta": "ikim-uk-essen/geberta-base"
}

datasets = {
    "clef": "/home/data/eval/clef",
}

# First run: Create hpset JSON files
for model_name, model_path in models.items():
    for dataset_name, dataset_path in datasets.items():
        max_seq_length = 128 if dataset_name == "ggponc2" else 64
        hpset = {
            "model_name_or_path": {"_type": "choice", "_value": [f"{model_path}"]},
            "dataset_name": {"_type": "choice", "_value": [f"{dataset_path}"]},
            "learning_rate": {"_type": "choice", "_value": [7e-05, 5e-05, 2e-05, 1e-05, 7e-06, 5e-06, 1e-06]},
            "per_device_train_batch_size": {"_type": "choice", "_value": [16, 32, 48, 64]}
        }
        output_dir = os.path.join("experiments", model_name, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"hpset_{model_name}_{dataset_name}.json")
        with open(output_file, "w") as f:
            json.dump(hpset, f, indent=4)

print("Hyperparameter sets created successfully.")

# Second run: Create config.yml files
for model_name, model_path in models.items():
    for dataset_name, dataset_path in datasets.items():
        config = f"""experimentName: {model_name}_{dataset_name}
searchSpaceFile: hpset_{model_name}_{dataset_name}.json
trialCommand: python run_classification.py
trialCodeDirectory: ../../../
trialGpuNumber: 1
trialConcurrency: {CUDA_VISIBLE_DEVICES}
tuner:
  name: GridSearch
  classArgs:
    optimize_mode: maximize
trainingService:
  platform: local
  maxTrialNumberPerGpu: 1
  useActiveGpu: true
"""
        output_dir = os.path.join("experiments", model_name, dataset_name)
        output_file = os.path.join(output_dir, "config.yml")
        with open(output_file, "w") as f:
            f.write(config)

print("Config files created successfully.")



