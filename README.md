# README

This repository contains an example of the code to load the [SeizeIT2 dataset](https://openneuro.org/datasets/ds005873) and to train the model included in the [dataset paper](https://arxiv.org/abs/2502.01224).

# loader_test.py
Script with an example for loading files from the dataset. The classes classes.data and classes.annotation are used to create a data object, containing the signal data and extra information,  and an annotation object, containing all information regarding the seizure events of the recording.

# main_net.py
Script to train and evaluate the ChronoNet model with all parameters as in the paper. This is a suggestion of a framework that uses the data loaders and a Keras implementation of the training and evaluation routines. The data generators are likely to take a long time to run (arround 3 hours), hence the option to save the training and validation generators and load them in future runs.

## Available Models

The repository supports four model architectures:

1. **ChronoNet** - CNN-GRU hybrid architecture with inception modules and residual connections
2. **EEGnet** - Compact convolutional neural network designed for EEG signal processing
3. **DeepConvNet** - Deep convolutional network adapted from Schirrmeister et al. (2017)
4. **TransformerEEG** - Transformer-based architecture using multi-head self-attention for EEG classification

To use a specific model, set `config.model` to one of: `'ChronoNet'`, `'EEGnet'`, `'DeepConvNet'`, or `'TransformerEEG'`.

### TransformerEEG Model

The TransformerEEG model uses a transformer encoder architecture with the following features:

- **Input format**: Accepts the same input shape as other models `(CH, frame*fs, 1)`, ensuring compatibility with existing preprocessing pipelines
- **Time tokenization**: Uses Conv1D patching to reduce sequence length and avoid O(T²) memory complexity
- **Architecture**: Multi-head self-attention layers with feed-forward networks, residual connections, and layer normalization
- **Output**: 2-class softmax classification compatible with existing loss functions and metrics

#### TransformerEEG Hyperparameters (optional, with defaults)

The TransformerEEG model uses default values if not specified in config:

- `transformer_patch_size` (default: 16) - Size of patches for time tokenization
- `transformer_patch_stride` (default: 8) - Stride between patches
- `transformer_embed_dim` (default: 64) - Embedding dimension
- `transformer_num_heads` (default: 4) - Number of attention heads
- `transformer_ff_dim` (default: 128) - Feed-forward network dimension
- `transformer_num_layers` (default: 2) - Number of transformer encoder layers
- `transformer_dropout_rate` (default: `config.dropoutRate`) - Dropout rate for transformer layers
- `transformer_max_pos_len` (default: 2048) - Maximum positional embedding length

Example usage:
```python
config.model = 'TransformerEEG'
# Optional: customize transformer hyperparameters
config.transformer_embed_dim = 128
config.transformer_num_layers = 4
```

## Conda environment setup
The python packages (and corresponding versions) used in the development of the scripts in this repository are gathered in 'environment.yml'. To easily create a conda environment with the same package versions to run the code, follow the instructions below:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda env create -n ENV_NAME -f environment.yml
```
