# NNI Evaluation for ChristBERT and Related Models

This project provides scripts and configuration files to automate hyperparameter search and evaluation for various German-language BERT models using the [NNI](https://github.com/microsoft/nni) (Neural Network Intelligence) toolkit. It supports both classification and named entity recognition (NER) tasks.

## Project Structure

```
├── cls/
│   ├── create_hpsets_configs.py   # Generates hyperparameter sets and NNI config files for classification
│   └── run_classification.py      # Runs classification experiments
├── ner/
│   ├── create_hpsets_configs.py   # Generates hyperparameter sets and NNI config files for NER
│   └── run_ner.py                 # Runs NER experiments
├── parse_best_hparams.py          # Parses best hyperparameters from NNI results
├── parse_best_metrics.py          # Parses best metrics from NNI results
├── parse_entity_metrics.py        # Parses entity-level metrics for NER
├── parse_results.py               # Parses general experiment results
├── parse_runtime.py               # Parses runtime information
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Main Features

- **Automated Hyperparameter Search:**
  - Generates JSON search spaces and NNI `config.yml` files for each model/dataset combination.
  - Supports grid search for learning rate and batch size.
- **Experiment Management:**
  - Organizes experiments in `experiments/<model>/<dataset>/` directories.
  - Compatible with NNI's local training service.
- **Result Parsing:**
  - Scripts to extract best hyperparameters, metrics, and runtime from NNI output.

## Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Hyperparameter Sets and Configs**
   - For classification:
     ```bash
     python cls/create_hpsets_configs.py
     ```
   - For NER:
     ```bash
     python ner/create_hpsets_configs.py
     ```

3. **Run NNI Experiments**
   - Use the generated `config.yml` files in `experiments/<model>/<dataset>/` with NNI:
     ```bash
     nnictl create --config experiments/<model>/<dataset>/config.yml
     ```

4. **Parse Results**
   - Use the provided parsing scripts to extract metrics and hyperparameters from NNI output.

## Requirements
- Python 3.8+
- NNI toolkit
- PyTorch, Transformers, and other dependencies (see `requirements.txt`)

## Notes
- Update dataset paths in `create_hpsets_configs.py` as needed.
- GPU device selection is set via `CUDA_VISIBLE_DEVICES` in the scripts.

## License
MIT.

## Contact
For questions or contributions, please contact the project maintainer.
