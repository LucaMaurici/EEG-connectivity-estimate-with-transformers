# A novel transformer-based approach for estimating causal interaction in multichannel electroencephalographic data

The "Thesis and Presentation" folder contains the thesis and Powerpoint presentation for my degree.
The "Spacetimeformer" folder provides access to the project code.
In the "connectivipy-master" folder is a useful library for estimating the connectivity used in my project.

In the project code, the most important folders are:
- data: contains both the EEG data used in the project and the python classes that handle this data in the pipeline.
- plots_checkpoints_logs: contains all the data generated during training and testing. It follows a precise structure that allows a large amount of data to be generated and stored automatically.
- spacetimeformer_model: contains the core code, which is strongly based on the original code from https://github.com/QData/spacetimeformer, except for some personal modifications and additions.
- statistical_test_attention and statistical_test_granger_causality: contain results from several statistical tests I performed.

Most of the file names are self-explanatory. An important fact to know is that the config.txt and config.csv files are used to set mostly common parameters between different types of training and testing runs.

EEG_connectivity_estimate_with_transformers.ipynb contains commands to launch train and test runs. I used it in a Colab notebook, to use GPU.

The rest of the README is a collection of the original README parts, from https://github.com/QData/spacetimeformer.


# Spacetimeformer Multivariate Forecasting (from the original README)

This repository contains the code for the paper, "**Long-Range Transformers for Dynamic Spatiotemporal Forecasting**", Grigsby, Wang and Qi, 2021. ([arXiv](https://arxiv.org/abs/2109.12218)). 

**Spacetimeformer** is a Transformer that learns temporal patterns like a time series model and spatial patterns like a Graph Neural Network.

**June 2022 disclaimer: the updated implementation no longer matches the arXiv pre-prints. We are working on a new version of the paper. GitHub releases mark the paper versions.**

Below we give a brief explanation of the problem and method with installation instructions. We provide training commands for high-performance results on several datasets.

## Data Format
We deal with multivariate sequence to sequence problems that have continuous inputs. The most common example is time series forecasting where we make predictions at future ("target") values given recent history ("context"):

![](readme_media/data_setup.png)

Every model and dataset uses this `x_context`, `y_context`, `x_target`, `y_target` format. X values are time covariates like the calendar datetime, while Ys are variable values. There can be additional context variables that are not predicted. 


## Spatiotemporal Attention
Typical deep learning time series models group Y values by timestep and learn patterns across time. When using Transformer-based models, this results in "*temporal*" attention networks that can ignore *spatial* relationships between variables.

In contrast, Graph Neural Networks and similar methods model spatial relationships with explicit graphs - sharing information across space and time in alternating layers.

Spactimeformer learns full spatiotemporal patterns between all varibles at every timestep.

![](readme_media/attention_comparison.png)

We implement spatiotemporal attention with a custom Transformer architecture and embedding that flattens multivariate sequences so that each token contains the value of a single variable at a given timestep:

![](readme_media/spatiotemporal_sequence.png)

Spacetimeformer processes these longer sequences with a mix of efficient attention mechanisms and Vision-style "windowed" attention.

![](readme_media/spacetimeformer_arch.png)

This repo contains the code for our model as well as several high-quality baselines for common benchmarks and toy datasets.


## Installation and Training
This repository was written and tested for **python 3.8** and **pytorch 1.11.0**.

```bash
git clone https://github.com/QData/spacetimeformer.git
cd spacetimeformer
conda create -n spacetimeformer python==3.8
source activate spacetimeformer
pip install -r requirements.txt
pip install -e .
```
This installs a python package called ``spacetimeformer``.


Commandline instructions for each experiment can be found using the format: ```python train.py *model* *dataset* -h```. 

Spacetimeformer has many configurable options and we try to provide a thorough explanation with the commandline `-h` instructions.


#### Datasets

    *(We load these benchmarks in an unusual format where the context sequence is *all data up until the current time* - leading to variable length sequences with padding.)*

### Logging with Weights and Biases
We used [wandb](https://wandb.ai/home) to track all of results during development, and you can do the same by providing your username and project as environment variables:
```bash
export STF_WANDB_ACCT="your_username"
export STF_WANDB_PROJ="your_project_title"
# optionally: change wandb logging directory (defaults to ./data/STF_LOG_DIR)
export STF_LOG_DIR="/somewhere/with/more/disk/space"
```
wandb logging can then be enabled with the `--wandb` flag.

There are several figures that can be saved to wandb between epochs. These vary by dataset but can be enabled with `--attn_plot` (for Transformer attention diagrams) and `--plot` (for prediction plotting).


## Example Training Commands

See EEG_connectivity_estimate_with_transformers.ipynb for commands to launch train and test runs.
