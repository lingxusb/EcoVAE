# EcoVAE: a deep learning approach for predicting species distributions
![github](https://github.com/user-attachments/assets/3432b7d5-ddcd-4a62-9639-be4674e05007)


- Anthropogenic pressures on biodiversity necessitate efficient and highly scalable methods to predict global species distributions. Current species distribution models (SDMs) face limitations with large-scale datasets, complex interspecies interactions, and data quality.
- We introduce **EcoVAE**, a framework of autoencoder-based generative models trained separately on nearly **124 million** georeferenced occurrences from taxa including plants, butterflies and mammals, to predict their global distributions at both genus and species levels.
- EcoVAE achieves high precision and speed, captures underlying distribution patterns through unsupervised learning, and reveals interspecies interactions via in silico perturbation analyses. Additionally, it evaluates global sampling efforts and interpolates distributions without relying on environmental variables, offering new applications for biodiversity exploration and monitoring. 


## Installation
Python package dependencies:
- torch 2.0.0
- pandas 1.3.4
- scikit-learn 0.24.2
- matplotlib 3.4.3
- tqdm 4.66.5

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

- ```train.txt```, which stores the training data for species distributions. The provided example file contains grid samples with 100 species. The value 0 indicates absence of the species and value 1 denotes existence of the species in the grid.
- ```test.txt```, which stores the observation data for the test dataset with 100 species.


To train the model, please run the following codes:

```
python training.py -e 15
```

For a full list of options, please run:

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

Output is the model file stored in ```./model/model.pth```.

The script should automatically detect whether to use CUDA (GPU) or CPU based on availability. If you encounter a CUDA-related error when running on a CPU-only machine, the script will handle this by falling back to CPU.

## Data interpolation

After training, you can apply the model to new data using the `interpolation.py` script. This script loads a trained model and applies it to new data. Please note that the new data should have the same number of species/genera as the training data.

- the trained model is provided in the ```./model/model_all_regions.pth```. The input and output dimension of the model is 13,125.
- the example dataset to interpolate is stored in ```./data/interpolation_input.npy```, which contains the observation data from 561 grids and with a shape of (561, 13125).


To run the script with default settings:

```
python interpolation.py
```

To specify a custom model and input file:

```
python interpolation.py --model custom_model.pth --input custom_input.txt
```
For a full list of options, please run:

```
python interpolation.py --help
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
## Species interactions
After training, you can apply the model to study the species interactions using the `interaction.py` script. This script loads a trained model and applies it to new data. Please note that the new data should have the same number of species/genera as the training data.

- the trained model is provided in ```./model/model_all_regions.pth```. The input and output dimension of the model is 13,125.
- the example dataset for studying the species interactions is stored in ```./data/interpolation_input.npy```, which contains teh observation data from 561 grids and with a shape of (561, 13125).
- the genus list is provided in ```./data/interaction/genus_list.npy```, which stores the names for 13,125 genera.

To run the script with default settings:

```
python interaction.py
```

For a full list of options, please run:

```
python interaction.py --help
```

The optional argument is similar to the ```interpolation.py``` file. The model output contains three files in the ```./data/interaction``` folder:
- ```interaction_genus.txt```, stores the names for all the genera that have been added in the grids,  length is *n*.
- ```interaction_background.txt```, size is (n, 13,125), row *i* stores the grid number (distribution range) for all the genera.
- ```interaction_addition.txt```, size is (n, 13,125), row *i* stores the predicted grid number (distribution range) for all the genera after addition of genus *i*.

## Model details
| Taxa      | plant | plant |butterfly |butterfly |mammal |mammal  |
|-----------|-------|---------|-------|---------|-------|---------|
|  level         | genus |  species  | genus |  species      | genus |  species      |
| species/genus number | 13k | 127k | 11k | 47k | 1k | 5k |
| occurrence records   | 34M | 34M  | 68M  | 66M  | 22M | 21M |
| model parameter | 7M | 65M | 6M | 24M | 1M | 3M |

## Reference
