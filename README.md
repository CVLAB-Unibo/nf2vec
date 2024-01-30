# nf2vec

This repository contains the code related to **nf2vec** framework, which is detailed in the paper [Deep Learning on 3D Neural Fields](https://arxiv.org/abs/2312.13277). In particular, here you can find the code regarding processing NeRFs. If you want to use the previous version of this framework for processing shapes, refer to [inr2vec](https://github.com/CVLAB-Unibo/inr2vec).


## MACHINE CONFIGURATION

Before running the code, ensure that your machine is properly configured. 
This project was developed with the following main dependencies:
* python==3.8.18
* torch==1.12.0+cu113
* torchvision==0.13.0+cu113
* nerfacc==0.3.5 (with the proper CUDA version set)
* wandb==0.16.0 

### nf2vec

What follows are commands that you can execute to replicate the environment in which *nf2vec* was originally trained:

1. Install Python 3.8.18:
    ```bash
    conda install python=3.8.18
    ```

2. Install pip:
    ```bash
    conda install -c anaconda pip
    ```

3. Install PyTorch and torchvision:
    ```bash
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

4. Install CUDA Toolkit:
    ```bash
    conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
    ```

5. Install Ninja and Tiny CUDA NN:
    ```bash
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```

6. Install NerfAcc:
    ```bash
    pip install nerfacc==0.3.5 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.12.0_cu113.html
    ```

7. Install Einops:
    ```bash
    conda install -c conda-forge einops
    ```

8. Install ImageIO:
    ```bash
    conda install -c conda-forge imageio
    ```

9. Install WanDB:
    ```bash
    pip install wandb==0.16.0
    ```
10. Install h5py:
    ```bash
    conda install -c anaconda h5py
    ```
11. Install TorchMetrics:
    ```bash
    pip install torchmetrics
    ```

### Generation
The generation task is based on a *Latent GAN* model detailed at [THIS](https://github.com/optas/latent_3d_points) link. Please, follow the instructions provided at that link to properly configure your environment.  

### Mapping Network
The mapping network task requires the training of the *inr2vec* framework. Please, refer to [THIS](https://github.com/CVLAB-Unibo/inr2vec?tab=readme-ov-file#setup) page to properly configure your environment.

## TRAINING AND EXPERIMENTS
This section contains the details required to run the code.

**IMPORTANT NOTES**: 
1. each module cited below *must* be executed from the root of the project, and not within the corresponding packages. This will ensure that all the paths used can properly work.

2. the file *settings.py* contains all the paths (e.g., dataset location, model weights, etc...) and generic configurations that are used from each module explained below. 

3. Some training and experiments, such as the training of the *nf2vec* framework and the classification task, use the *wandb* library. If you want to use it, then you need to change the following two variables: ``` os.environ["WANDB_SILENT"]``` and  ```os.environ["WANDB_MODE"]```, which are located at the beginning of the *settings.py* module. 

## Train *nf2vec*

To train *nf2vec* you need to have a dataset of trained NeRFs. The implemented code expects that there exist the following files:
* data/train.json
* data/validation.json
* data/test.json

These JSONs hold a list of file paths, with each path corresponding to a NeRF model that has been trained, and then used in a specific data split. In particular, each path corresponds to a folder, and each folder contains the following relevant files:
* the trained NeRF's weights
* the NeRF's occupancy grid
* JSON files with transform matrices and other paramters necessary to train NeRFs.

The name of the files contained in these folders should not be changed. Within the repository, you can find the JSON files used to originally train the framework.

Execute the following command to train *nf2vec*:
```bash
python nerf2vec/train_nerf2vec.py
```
If you have enabled *wandb*, then you should update its settings located in the *config_wandb* method, which is localed in the *train_nerf2vec.py* module.

## Export *nerf2vec* embeddings
Execute the following command to export the *nerf2vec*'s embeddings:
```bash
python nerf2vec/export_embeddings.py
```
Note that these embeddings are **necessary** for other tasks, such as classification, retrieval and generation.

## Retrieval task
Execute the following command to perform the retrieval task:
```bash
python task_interp_and_retrieval/retrieval.py
```
The results will be shown in the *task_interp_and_retrieval/retrieval_plots_X* folder, where X depends on the chosen split (i.e., train, validation or test). The split can be set in the *main* method of the *retrieval.py* module.

Each file created during a specific retrieval iteration will be named using the same prefix represented by a randomly generated UUID.


## Interpolation task
Execute the following command to perform the interpolation task:
```bash
python task_interp_and_retrieval/interp.py
```
The results will be shown in the *task_interp_and_retrieval/interp_plots_X* folder, where X depends on the chosen split (i.e., train, validation or test). The split can be set in the *main* method of the *retrieval.py* module.

## Classification task
Execute the following command to perform the classification task:
```bash
python task_classification/train_classifier.py
```
If you have enabled *wandb*, then you should update its settings located in the *config_wandb* method, which is localed in the *train_classifier.py* module.

## Generation task
In order to generate and visualize the new embeddings, it is necessary to execute some operations following a specific order.

### 1) Export embeddings
The following command creates the folder *task_generation/latent_embeddings*, which will contain the *nerf2vec*'s embedding properly organized for this task.
```bash 
python task_generation/export_embeddings.py
```

### 2) Train GANs
The following command creates the folder *task_generation/experiments*, which will contain both the weights of the trained models and the generated embeddings:
```bash
python task_generation/train_latent_gan.py
```
All the hyperparameters used to train the *Latent GANs* can be found inside the *train_latent_gan.py* module.

### 3) Create renderings
The following command creates renderings from the embeddings generated during the previous step:
```bash
python task_generation/viz_nerf.py 
```
The renderings will be created in the *GAN_plots_X* folder, where X is the ID of a specific class.

## Mapping network map
Please refer to [THIS](task_mapping_network/README.md) README for this task.

# Datasets and model weights
Please contact us if you need access to the datasets, exported embeddings, and weights of the trained models used in all experiments.
