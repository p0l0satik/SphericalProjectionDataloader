# SphericalProjection
Plane segmentation pipeline with spherical projection

# Data preparation

## Downloading dataset
[Download](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) Semantic Kitti dataset. 
You will need sequence **00**.

[Download](https://drive.google.com/drive/folders/17qZpTi3BYhQTHxlMwsNHf5QK8rCpmocL?usp=sharing)
EVOPS plane labels for Semantic Kitti dataset.

Put **velodyne** and **labels**  as well as evops labels renamed to **plane_labels** into one directory. 
In the end the directory with data should look like this.
```
├── labels
│   └── 000000.label
├── plane_labels
│   └── label-000000.npy
└── velodyne
    └── 000000.bin
```
You can find an example in `examples/data/kitti/00`. 

## Preparing dataset
The next step is running data preprocessing script. This can be done with the `preprocessing.py` script.
It takes the following parameters:
##### Required
- `--dataset` path to your dataset folder
- `--dataset_len` length of your dataset
- `--save_path` path to store ready to use data

#### Optional
By default, preprocessing applies ransac filtering and considers road labels as planes. However, you can change that.
- `--no_ransac` turns off ransac filtering
- `--no_road` road labels will not be considered as planes
- `--visualise_ransac` ransac filtering steps will be visualised

An example of preprocessing script run, applied to the data in **examples** directory:
> python3 preprocessing.py --dataset examples/data/kitti/00/ 
> --save_path examples/data/dataset_ready/ --dataset_len 1

# Training network

A train run is configured with a config file in a `.yaml` format. You can find an example in 
**runs_configs** directory. If you wish to train the network with your own parameters create a copy of an existing run 
and change the parameters you like. 

This project uses [wandb](https://wandb.ai/site) for logging data during train. In order to use it too you need:
1. Log into your wandb account
2. Set `use_wandb` to **True** in config 
3. Fill in `wandb_project` with your project name
In case you don't want to use wandb train pipline will log with the help of console.

In order to run train successfully you also need to fill the following variables in config:

- `dataset_dir` - the path to your **_prepared_** dataset
- `checkpoint_save_path` - the path where to store your network checkpoints

After the run is configured you can run train with `run_network.py`. 
This script takes only one required parameter - path to your config file. Here is an example:

> python3 run_network --config runs_configs/exp1_basic.yaml

# Visualising results

The results can be visualized with `visualise_prediction.py`. It takes the following parameters.
##### Required
- `--chpt` path to your network checkpoint
- `--dataset` path to the dataset you used to train your network
- `--config` path to the config you used for network training

#### Optional
- `--device` by default the network prediction is run on **cpu** for visualizing results on any device. You can change this.
- `--step` you can skip similar frames of the dataset configuring step. Default value is 5.
- `--iterations` number of samples you wish to visualise. Default is 1.

An example of visualising script run will produce 3 visualisations (15 samples with step 5) :
> python3 visualise_prediction.py --chpt exp_2_basic_unet_ransac_14 
> --dataset dataset/kitti/prep/ransac_01 --iterations 15