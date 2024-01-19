# Task Mapping Network
The mapping network task requires the training of the *inr2vec* framework. Please, refer to [THIS](https://github.com/CVLAB-Unibo/inr2vec?tab=readme-ov-file#setup) page to properly configure your environment.

In order to complete this task, it is necessary to execute some operations following a specific order.

## 1) Create point clouds
This step is necessary to create the dataset on which *inr2vec* will be trained. It is important to update the variable *shapenet_root* found in *task_mapping_network/cfg/pcd_dataset.yaml*. This variable should point to the root of the *ShapeNet* folder.

Then, execute the following command:
```bash
python task_mapping_network/inr2vec/create_point_clouds_dataset.py 
```

## 2) Create INRs dataset
Create the INRs dataset by executing the following command:
```bash
python task_mapping_network/inr2vec/create_inrs_dataset.py
```
The file *task_mapping_network/cfg/inrs_dataset.yaml* contains all the configurations used for this step.

## 3) Train *inr2vec*
Train *inr2vec* with the following command:
```bash
python task_mapping_network/inr2vec/train_inr2vec.py
```
The file *task_mapping_network/cfg/inr2vec.yaml* contains all the configurations used for this step.

## 4) Export *inr2vec* and *nerf2vec* embeddings
Create embeddings that will be properly organized to train the mapping network:
```bash
python task_mapping_network/export_inrs_embeddings.py
python task_mapping_network/export_nerfs_embeddings.py
```

The file *task_mapping_network/cfg/export_embeddings.yaml* contains all the configurations used for this step.

## 5) Train the mapping network
Train the mapping network:
```bash
python task_mapping_network/train_completion.py
```
The file *task_mapping_network/cfg/completion.yaml* contains all the configurations used for this step.


## 6) Export results
Visualize the results by executing:
```bash
python task_mapping_network/viz.py
```
The file *task_mapping_network/cfg/completion.yaml* contains all the configurations used for this step.

The results will be saved in the *task_mapping_network/completion_plots* folder.

