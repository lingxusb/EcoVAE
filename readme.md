# EcoVAE: a deep learning approach for predicting species distributions

Here we introduce EcoVAE, a deep learning model to predict and interpolate global species distributions using incomplete input data. Trained on 33.8 million specimen occurrence records, EcoVAE demonstrates high precision in predicting full species distributions in regions withheld from the training data. It effectively interpolates species presence in areas with both abundant and sparse samples, while also uncovering intrinsic species interactions that may not be directly observable in original records.

## Installation
Python package dependencies:
- torch 2.0.0
- pandas 1.3.4
- scikit-learn 0.24.2
- matplotlib 3.4.3

We recommend using [Conda](https://docs.conda.io/en/latest/index.html) to install our packages. For convenience, we have provided a conda environment file with package versions that are compatiable with the current version of the program. The conda environment can be setup with the following comments:

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

## Model training

The following codes could be used to train the EcoVAE model to predict the species distribution from partial observations. The corresponding training and test datasets are provided in the ```./data``` folder.

- ```train.txt```, which stores the training data for species distributions. The provided example file contains 1000 grid samples with 100 species. The value 0 indicates absence of the species and value 1 denotes existence of the species in the grid.
- ```test.txt```, which stores the observation data for the test dataset. The provided file contains 200 sample grids.


To train the model, please run the following codes:

```
python training.py -e 15
```

For a full list of options, run:

```
python training.py --help
```
The full command is:
```
python  training.py [-h] [-e NUM_EPOCHS] [-hd HIDDEN_DIM] [-ld LATENT_DIM] [-l LEARNING_RATE] [-b BATCH_SIZE] [-m MASKING_RATIO] [-lw LAMBDA_WEIGHT]
```

Optional arguments

| argument | description |
| ------------- | ------------- |
| ```-h```, ```--help```  | show help message and exit  |
|  ```-e```, ```--num_epochs```| number of epochs to train the model, default value: 15  |
| ```-hd```, ```--hidden_dim```| hidden dimension for the encoder and decoder, default value: 256  |
| ```-ld```, ```--latent_dim``` | latent dimension, default value: 32  |
| ```-l```, ```--learning_rate``` | learning rate for model training, default value: 1e-3 |
| ```-b```, ```--batch_size``` | batch size for model training, default value: 512  |
| ```-m```, ```--masking_ratio``` | the ratio of the input data that are randomly masked in the model training process, default value: 0.5 |
| ```-lw```, ```--lambda_weight``` | the weight to balance the reconstruction loss for the masked and unmasked data, default = 0.5  |

Model output is the model file stored in ```./model/model.pth```.

The script should automatically detect whether to use CUDA (GPU) or CPU based on availability. If you encounter a CUDA-related error when running on a CPU-only machine, the script will handle this by falling back to CPU.

## Data interpolation

After training, you can apply the model to new data using the `interpolation.py` script. This script loads a trained model and applies it to new data from a text file. Please note that the new data should have the same number of species/genera as the training data.

- the trained model is provided in the ```./model/model_all_regions.pth```. The input and output dimension of the model is 13,125.
- the example dataset to interpolate is stored in ```./data/interpolation_input.npy```, which contains teh observation data from 561 grids and with a shape of (561, 13125).


To run the script with default settings:

```
python interpolation.py
```

To specify a custom model and input file:

```
python interpolation.py --model custom_model.pth --input custom_input.txt
```
For a full list of options, run:

```
python training.py --help
```

Optional arguments

| argument | description |
| ------------- | ------------- |
| ```-h```, ```--help```  | show help message and exit  |
|  ```-m```, ```--model```| model file name, default value: model_all_regions.pth  |
| ```-i```, ```--input```| input file name, default value: interpolation_input.npy  |
| ```-o```, ```--output``` | output file name, default value: interpolation_results.npy  |
| ```-id```, ```--input_dim``` | input dimension of the model, default value: 13125 |
| ```-hd```, ```--hidden_dim``` | dimension of the hidden layer, default value: 256  |
| ```-ld```, ```--latent_dim``` | dimension of the latent space, default value: 32 |
| ```-b```, ```--batch_size``` | batch size for model inference, default value: 2048  |

Model output are predicted scores for species occurrence which has the same size with the model input.

## Project Structure

- `env.yml`: Conda environment file
- `training.py`: Main training script
- `model.py`: VAE model definition
- `data_loader.py`: Data loading utilities
- `utils.py`: Utility functions
- `interpolation.py`: Script to apply trained model to new data
- `README.md`: Readme file

## Reference
