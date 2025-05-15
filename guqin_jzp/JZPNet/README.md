# JZPNet: Jianzipu Optical Music Recognition

JZPNet is a deep learning project for Optical Music Recognition (OMR) of Jianzipu (减字谱), a tablature notation for the Chinese musical instrument Guqin. It utilizes PyTorch Lightning for training and evaluation.

## Project Structure

dataset/              # Raw dataset files (images, annotations)
├── dataset.json      # Main dataset annotations
├── edge_dict.txt     # Vocabulary for graph edges
├── metadata.txt      # General metadata
└── node_dict.txt     # Vocabulary for graph nodes
JZPNet/
├── jzp.yml           # Example configuration file
├── main.py           # Main script for training, testing, and k-fold validation
├── README.md         # This file
├── requirements.txt  # Python dependencies
├── testJZP.sh        # Shell script with example commands
├── data/             # Processed data, vocabularies, and specific dataset splits
│   ├── jzp/          # Example dataset directory
│   │   └── metadata_3.txt # Specific metadata for this dataset split
│   ├── jzpdata/
│   │   ├── dataset.json     # Original dataset annotations (potentially duplicated or linked)
│   │   └── gui_config.json  # Configuration for character extraction
│   ├── wushen_jzp/
│   └── zyt/
├── datamodules/      # PyTorch Lightning DataModules and data utilities
│   ├── jianzipu.py   # Defines JZPDataset and JZPDataModule
│   ├── read_jzp.py   # Utilities for reading and parsing Jianzipu data from dataset.json
│   ├── create_tree.py # (Purpose to be inferred or documented)
│   └── utils.py      # General data utilities
├── models/           # Model definitions
│   ├── tad_module.py # Defines TSDNetModule (Lightning wrapper)
│   ├── tadnet.py     # Defines TADNet (core network architecture)
│   ├── densenet.py   # Densenet components
│   ├── embedding.py  # Embedding layers
│   ├── encoder.py    # Encoder components
│   ├── pointer.py    # Pointer network components
│   ├── positional.py # Positional encoding
│   ├── tadlayer.py   # Layers specific to TADNet
│   ├── tsa.py        # (Purpose to be inferred or documented, possibly Transformer Self-Attention)
│   └── utils.py      # Model utilities
└── preprocess/       # Scripts for data preprocessing and visualization
    └── get_string.py # Script to visualize model prediction on a single image

## Setup

### Prerequisites

*   Python 3.10
*   pip

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>/JZPNet
    ```

2.  **Install dependencies:**
    Install the required Python packages using the requirements.txt file:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include:
    *   `editdistance==0.5.3`
    *   `einops==0.3.0`
    *   `numpy==1.18.5`
    *   `pandas==1.1.4`
    *   `Pillow==9.1.0`
    *   `pytorch_lightning==1.5.8`
    *   `torch==1.10.1`
    *   `torchmetrics==0.6.2`
    *   `torchvision==0.11.2`
    *   `tqdm==4.62.3`
    *   `pyyaml` (implicitly required by PyTorch Lightning for config files)

## Dataset

The project processes Jianzipu image data and their corresponding annotations.

*   **Raw Annotations**: The primary raw dataset annotations are expected in `dataset/dataset.json`.
*   **Vocabularies**:
    *   `dataset/node_dict.txt`: Contains the vocabulary for nodes in the Jianzipu representation.
    *   `dataset/edge_dict.txt`: Contains the vocabulary for edges.
    These are also typically copied or linked to `JZPNet/data/jzp/` for use by the datamodules.
*   **Processed Data**:
    *   Image files and specific metadata are organized into subdirectories within `JZPNet/data/` (e.g., `JZPNet/data/jzp/`, `JZPNet/data/wushen_jzp/`).
    *   Each dataset subdirectory (e.g., `JZPNet/data/jzp/`) is expected to contain a `metadata_3.txt` file, which is read by the `JZPDataset` class in the `datamodules/jianzipu.py` file.
*   **Data Loading**:
    *   The `datamodules/jianzipu.py` file defines `JZPDataset` for individual dataset instances and `JZPDataModule` for managing training, validation, and test data loaders.
*   **Data Parsing Utilities**:
    *   The `datamodules/read_jzp.py` script provides functions to parse `data/jzpdata/dataset.json`:
        *   `parse_annotation`: Parses different types of annotations.
        *   `parse_full_jianzipu`: Specifically parses full Jianzipu content.
        *   `get_jzp_string`: Extracts string representations of annotations and image paths.
        *   `get_jzp_character`: Extracts a list of unique Jianzipu characters using `data/jzpdata/gui_config.json`.
        *   `get_jzp_structure`: Extracts structural symbols from Jianzipu notations.

## Usage

The main script for interacting with the model is `main.py`. It supports several commands:

*   `train_test`: Train the model and then test it.
*   `test`: Test a pre-trained model.
*   `kfold`: Perform k-fold cross-validation. The `kfold` function in `main.py` initializes `JZPDataModule` with `n_split` (e.g., 5 folds) and iterates through each fold for training and validation.

### Configuration

Model and training parameters are specified in a YAML configuration file (e.g., `jzp.yml`). This file is passed to the main script using the `--config` argument.

### Examples

The `testJZP.sh` script provides examples of how to run the different modes.

**Train and Test:**
```bash
python main.py train_test --dataset jzp --log_dir final_logs --exp_name jzp --config "configs/jzp.yml" --gpu 0 --progress_bar
```

**Test Only:**
```bash
python main.py test --dataset jzp --log_dir final_logs --exp_name jzp_test --config "configs/jzp.yml" --gpu 0 --ckpt_path "path/to/checkpoint.ckpt"
```

**K-Fold Cross Validation:**
```bash
python main.py kfold --dataset jzp --log_dir kfold_logs --exp_name jzp_kfold --config "configs/jzp.yml" --gpu 0 --n_split 5
```

## Model Architecture

JZPNet is based on a Transformer-Aware Decoder Network (TADNet) architecture. The model consists of:

1. **DenseNet Image Encoder**: Extracts visual features from Jianzipu notation images
2. **Transformer-based Encoder**: Processes the extracted features with self-attention mechanisms
3. **Tree-Structured Decoder**: Generates a hierarchical representation of the music notation

The model architecture is defined in the `models/` directory, with the core implementation in `tadnet.py`.

## Evaluation Metrics

The model performance is evaluated using multiple metrics:

- **Character Error Rate (CER)**: Measures the accuracy of recognized Jianzipu symbols
- **Structure Recognition Accuracy**: Evaluates the model's ability to correctly identify the hierarchical structure of the notation
- **Graph Edit Distance**: Quantifies the difference between predicted and ground truth graph representations

## License

This project is licensed under [LICENSE NAME] - see the LICENSE file for details.

## Contributing

Contributions to improve JZPNet are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
