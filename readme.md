# VAE Training Project

This project implements a Variational Autoencoder (VAE) for training on masked data and applying the trained model to new data.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/lingxusb/mvae.git
   cd mvae
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f env.yml
   conda activate mvae
   ```


## Training

To train the model, run:

```
python training.py -e 15
```

You can adjust other hyperparameters using command-line arguments. For a full list of options, run:

```
python training.py --help
```

## Applying the Trained Model

After training, you can apply the model to new data using the `apply_model.py` script. This script loads a trained model and applies it to input data from a text file.

### Usage

To run the script with default settings:

```
python interpolation.py
```

This will use "model_all_regions.pth" as the model and "input_data.txt" as the input file.

To specify a custom model and input file:

```
python interpolation.py --model custom_model.pth --input custom_input.txt
```

### Notes

- The script automatically detects whether to use CUDA (GPU) or CPU based on availability.
- If you encounter a CUDA-related error when running on a CPU-only machine, the script should handle this automatically by falling back to CPU.
- Results are saved to "output_results.txt" by default.

## Project Structure

- `env.yml`: Conda environment file
- `training.py`: Main training script
- `model.py`: VAE model definition
- `data_loader.py`: Data loading utilities
- `utils.py`: Utility functions
- `interpolation.py`: Script to apply trained model to new data
- `README.md`: This file

## License

MIT license

