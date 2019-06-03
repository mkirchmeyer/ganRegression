# Regression with CGAN Code
This repository contains the code of the following paper 

K Aggarwal, M Kirchmeyer, P Yadav, S Sathiya Keerthi, P Gallinari, "[Regression with Conditional GAN](https://arxiv.org/abs/1905.12868)"

# Dependencies
In order to run, the code requires the following Python modules referenced in `requirements.txt`:
  * numpy, jupyter, matplotlib, pandas
  * sklearn
  * tensorflow, keras
  * GPy `https://sheffieldml.github.io/GPy/`
  
CGAN code is derived from `https://github.com/eriklindernoren/Keras-GAN`

# Quickstart
* Create conda environment: `conda create --name ganRegression python=3.6 -y` 
* Install the requirements in this environment `pip install -r requirements.txt`
* Install the package `pip install -e .` at the root
* Run the notebooks using `jupyter-notebook`

# Notebooks
* Run `notebook/synthetic_data.ipynb` for synthetic data
* Run `notebook/real_world_data.ipynb` for real world data
* Notebooks will save figures in the `figures` folder for each data scenario

# Datasets
* Synthetic datasets: `linear`, `sinus`, `heteroscedasitic`, `exp`, `multi-modal`
* Real World datasets: `CA-housing`, `ailerons`. `CA-housing-single` takes the most important feature from `CA-housing` 
(cf. study in the paper)

# Config
* The Config class handles all parameters. These are set at the beginning of each notebook. Refer to `config.py` for more details
* Architectures are fixed in `cgan_model.py` or can be set in the Config object for custom experiments.

# Results and uncertainty
* Results from the paper can be reproduced with an uncertainty smaller than 0.05 on NLPD + MAE for CGAN.