# EcoVAE: a generative deep learning approach for predicting species distributions
<img width="4286" height="2564" alt="github" src="https://github.com/user-attachments/assets/6480b0ec-0acc-4542-a8e1-3c8e8395f17d" />




- Anthropogenic pressures on biodiversity necessitate efficient and highly scalable methods to predict global species distributions. Current species distribution models (SDMs) face limitations with large-scale datasets, complex interspecies interactions, and data quality.
- We introduce **EcoVAE**, a framework of autoencoder-based generative models that integrates bioclimatic variables with georeferenced occurrences and is trained separately on nearly **124 million** georeferenced occurrences from taxa including plants, butterflies and mammals, to predict their global distributions at both genus and species levels.
- We present two versions of EcoVAE: one learns the patterns of global species distributions using occurrences only (**EcoVAE-o**); the other uses occurrences and climatic variables together (**EcoVAE-c**). 
- EcoVAE achieves high precision and speed, captures underlying distribution patterns through unsupervised learning, and reveals interspecies interactions via in silico perturbation analyses. Additionally, it evaluates global sampling efforts and interpolates distributions without relying on environmental variables, offering new applications for biodiversity exploration and monitoring.

  
## Table of Contents

- [Installation](#Installation)
- [Tutorial](#Tutorial)
- [Model training](#Model-training)
- [Spatial Block Cross Validation](#Spatial-Block-Cross-Validation)
- [Benchmarking against conventional SDMs](#Benchmarking-against-conventional-SDMs)
- [Data interpolation](#Data-interpolation)
- [Species interactions](#Species-interactions)
- [Model details](#Model-details)

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
   git clone https://github.com/lingxusb/EcoVAE.git
   cd EcoVAE
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f env.yml
   conda activate ecovae
   ```

## Tutorial
We provide a [tutorial notebook](https://github.com/lingxusb/EcoVAE/blob/main/notebooks/tutorial.ipynb) that walks through the entire process, including data processing from raw GBIF output, model training, prediction error calculation, and plotting of prediction error. The example dataset that contains 259k processed occurrence records can be downloaded from GBIF (https://www.gbif.org/occurrence/download/0000223-250117142028555). We also provided the R script for data processing in ```./data/tutorial/quick_data_clean_cluster_upload.R```

Below we provide details for each function.


## Model training

The following codes could be used to train the EcoVAE model to predict the species distribution from partial observations. We provide two training modes:

- **Presence mode** (`--mode presence`): Uses only species occurrence data for training
- **Climate mode** (`--mode climate`): Integrates both species occurrence data and climate variables for enhanced predictions

The example training and test datasets are provided in the ```./data``` folder:

- ```train.txt```, which stores the training data for species distributions. The provided example file contains grid samples with 100 species. The value 0 indicates absence of the species and value 1 denotes existence of the species in the grid.
- ```test.txt```, which stores the observation data for the test dataset with 100 species.


To train the model, please run the following codes:

```
python training.py -e 15
```

To train the model in climate-enhanced mode:

```
python training.py -e 15 --mode climate --climvar 19
```

For a full list of options, please run:

```
python training.py --help
```
The full command is:
```
python training.py [-h] [-e NUM_EPOCHS] [-hd HIDDEN_DIM] [-ld LATENT_DIM] [-l LEARNING_RATE] [-b BATCH_SIZE] [-m MASKING_RATIO] [-lw LAMBDA_WEIGHT] [-kl KL_WEIGHT] [-a {relu,gelu,elu}] [-i INPUT] [-o OUTPUT] [--mode {presence,climate}] [--climvar CLIMVAR]
```

Optional arguments

| argument | description |
| ------------- | ------------- |
| ```-h```, ```--help```  | show help message and exit  |
| ```-e```, ```--num_epochs```| number of epochs to train the model, default value: 20  |
| ```-hd```, ```--hidden_dim```| hidden dimension for the encoder and decoder, default value: 2048  |
| ```-ld```, ```--latent_dim``` | latent dimension, default value: 32  |
| ```-l```, ```--learning_rate``` | learning rate for model training, default value: 1e-3 |
| ```-b```, ```--batch_size``` | batch size for model training, default value: 512  |
| ```-m```, ```--masking_ratio``` | the ratio of the input data that are randomly masked in the model training process, default value: 0.5 |
| ```-lw```, ```--lambda_weight``` | the weight to balance the reconstruction loss for the masked and unmasked data, default value: 0.5  |
| ```-kl```, ```--kl_weight``` | KL divergence weight, default value: 0.0  |
| ```-a```, ```--activation``` | activation function (relu, gelu, elu), default value: relu  |
| ```-i```, ```--input``` | input file name, read the toy dataset if not given  |
| ```-o```, ```--output``` | output file name, default value: ./model/model.pth  |
| ```--mode``` | training mode: presence-only or climate-enhanced, default value: presence  |
| ```--climvar``` | number of climate variables (used in climate mode), default value: 19  |

Output is the model file stored in ```./model/model.pth```.

The script should automatically detect whether to use CUDA (GPU) or CPU based on availability. If you encounter a CUDA-related error when running on a CPU-only machine, the script will handle this by falling back to CPU.


## Spatial Block Cross Validation

To evaluate model performance with geographic generalization, we provide a spatial block cross-validation benchmarking tool. This approach divides the geographic space into blocks and ensures that training and testing data come from different spatial regions, providing a more robust assessment of model generalization to new geographic areas.

The benchmarking script performs 5-fold spatial block cross-validation on species distribution data. Example benchmark datasets are provided in the ```./data/benchmarking/``` folder:

- ```benchmark_input.npy```: Species occurrence data with shape (n_samples, n_species)
- ```benchmark_coords.npy```: Geographic coordinates with shape (n_samples, 2) containing [longitude, latitude]

To run spatial cross-validation with default settings:

```
python benchmarking.py
```

To specify custom data files and parameters:

```
python benchmarking.py --results your_data.npy --coords your_coords.npy --n_folds 5 --block_size 20 --margin 1
```

For a full list of options, please run:

```
python benchmarking.py --help
```

| argument | description |
| ------------- | ------------- |
| ```--results``` | path to species occurrence data (npy file), default: ./data/benchmarking/benchmark_input.npy |
| ```--coords``` | path to coordinate data (npy file), default: ./data/benchmarking/benchmark_coords.npy |
| ```--n_folds``` | number of cross-validation folds, default: 5 |
| ```--block_size``` | size of spatial blocks in degrees, default: 20 |
| ```--margin``` | margin size in degrees for buffer between train/test regions, default: 1 |
| ```--seed``` | random seed for reproducibility, default: 42 |
| ```--hidden_dim``` | hidden dimension for encoder and decoder, default: 2048 |
| ```--latent_dim``` | latent dimension, default: 32 |
| ```--batch_size``` | batch size for training, default: 2048 |
| ```--learning_rate``` | learning rate, default: 1e-3 |
| ```--masking_ratio``` | ratio of input data randomly masked during training, default: 0.5 |

Model performance will be saved in csv files in the ```./data/benchmarking/``` folder.


## Benchmarking against conventional SDMs

This pipeline generates species distribution models (SDMs) using the `biomod2` package in R, based on presence data of a single species/genus and bioclimatic variables. The workflow is designed to run in parallel on a computing cluster for efficient large-scale modeling.

- To run the model for a specific species (e.g., *Orthocarpus*), use the following command:
  
```
sbatch biomod_run_cluster_genus.sh "Orthocarpus" "1"
```

where:
  - `"Orthocarpus"` is the target genus or species name.
  - `"1"` specifies the grid to be used as the test region for model evaluation.

- The modeling script (`biomod_single_genus_by_grid.R`) uses bioclimatic variables as environmental predictors. These variables are automatically cropped and filtered based on the species’ range and the number of presence records, allowing for tailored and efficient model training.

- The modeling process includes multiple algorithms:
  - MaxENT
  - Downsampled Random Forest
  - Generalized Linear Models (GLM)
  - An ensemble model that combines individual models with AUROC > 0.8 using a weighted average approach

- After completing all species models, run the following script to generate a summary table of the results:

  ```
  Rscript summarize_output_genus.R
  ```

  The output includes:
  - Area Under the ROC Curve (AUROC)
  - True Skill Statistic (TSS)
  - Variable importance scores
  - Processing time
  - Model performance summaries for all species

All related codes are provided in the ```./SDM/``` folder.


## Data interpolation

After training, you can apply the model to new data using the `interpolation.py` script. This script loads a trained model and applies it to new data. Please note that the new data should have the same number of species/genera as the training data.

- the trained model is provided in the ```./model/model_all_regions.pth```. The input and output dimension of the model is 11,555 (plant distribution model at the genus level). The full model can be downloaded from [google drive](https://drive.google.com/file/d/16g2TfTnxDwCqvk-MsLiNz8bvowgfGzQM/view?usp=drive_link) 
- the example dataset to interpolate is stored in ```./data/interpolation_input.npy```, which contains the observation data from 561 grids and with a shape of (561, 11,555).


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
| ```-id```, ```--input_dim``` | input dimension of the model, default value: 11555 |
| ```-hd```, ```--hidden_dim``` | dimension of the hidden layer, default value: 256  |
| ```-ld```, ```--latent_dim``` | dimension of the latent space, default value: 32 |
| ```-b```, ```--batch_size``` | batch size for model inference, default value: 2048  |

Model output are predicted scores for species occurrence which has the same size with the model input.

We also provided a [jupyter notebook](https://github.com/lingxusb/EcoVAE/blob/main/notebooks/interpolation.ipynb) for data interpolation.


## Species interactions
After training, you can apply the model to study the species interactions using the `interaction.py` script. This script loads a trained model and applies it to new data. Please note that the new data should have the same number of species/genera as the training data.

- the trained model is provided in ```./model/model_all_regions.pth```. The input and output dimension of the model is 11,555 (plant distribution model at the genus level). The full model can be downloaded from [google drive](https://drive.google.com/file/d/16g2TfTnxDwCqvk-MsLiNz8bvowgfGzQM/view?usp=drive_link) 
- the example dataset for studying the species interactions is stored in ```./data/interpolation_input.npy```, which contains the observation data from 561 grids and with a shape of (561, 11,555).
- the genus list is provided in ```./data/interaction/genus_list.npy```, which stores the names for 11,555 plant genera.

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
- ```interaction_background.txt```, size is (n, n), row *i* stores the grid number (distribution range) for all the genera.
- ```interaction_addition.txt```, size is (n, n), row *i* stores the predicted grid number (distribution range) for all the genera after addition of genus *i*.

We provide a [jupyter notebook](https://github.com/lingxusb/EcoVAE/blob/main/notebooks/interaction.ipynb) for analyzing the model output and [another one](https://github.com/lingxusb/EcoVAE/blob/main/notebooks/mammal_interaction.ipynb) to reproduce Figure 5 d-f.

## Model details
| Taxa      | plant | plant |butterfly |butterfly |mammal |mammal  |
|-----------|-------|---------|-------|---------|-------|---------|
|  level         | genus |  species  | genus |  species      | genus |  species      |
| species/genus number | 12k | 127k | 11k | 47k | 1k | 5k |
| occurrence records   | 34M | 34M  | 68M  | 66M  | 22M | 21M |
| model parameter | 129M | 448M | 42M | 155M | 0.70M | 23M |

Our models are available from [HuggingFace](https://huggingface.co/lingxusb/EcoVAE/tree/main). Our training data are publicly available from GBIF (https://www.gbif.org/). Please check our [preprint](https://www.biorxiv.org/content/10.1101/2024.12.10.627845v1) for more details.


## Reference
- [A generative deep learning approach for global species distribution prediction](https://www.biorxiv.org/content/10.1101/2024.12.10.627845v1)
