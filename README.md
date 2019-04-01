# taylor-pruning

Reproducing ICLR'17 paper _Pruning Convolutional Neural Networks for Resource Efficient Inference_ ([link](https://arxiv.org/abs/1611.06440)).

This project is built on PyTorch.

## Install

Packages required are managed by Anaconda.

```shell
# install required packages
conda env create -f environment.yml
conda activate taylor-pruning

# install this repository
pip install -e .
```

## Usage

The project is still in progress but you may experiment with the following commands.

### Transfer a model

Download an ImageNet pre-trained model from TorchVision and fit its weights to the target dataset.

```shell
python transfer.py -a vgg16 --pretrained -d cub200 --dataset-dir $DATASET -c checkpoints/vgg16 --epochs 60 --lr 1e-4
```

### Prune a model

Run the pruning loop proposed by the paper.

```shell
python prune.py -a vgg16 -d cub200 --dataset-dir $DATASET --resume checkpoints/vgg16 -c checkpoints/pruning/vgg16 --epochs 30 --lr 1e-4 --wd 1e-4 --num-prune-iters 10 --num-channels-per-prune 100
```

This command runs 10 pruning iterations, and in each iteration, it prunes 100 channels of activations.

Results will be saved under the checkpoint directory provided.

### Development and Troubleshooting

Please ask any problem you have through issues.
