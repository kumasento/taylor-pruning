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

### Data preprocessing

We can accept images from ImageNet (`imagenet`) and CUB-200 (`cub200`) datasets.

The directory of dataset should contain two sub-directories named as `train` and `val`. In either of them, all samples should be organised into subdirectories named as the labels.

Please refer to the `ImageFolder` class in PyTorch for more information.

Input images are transformed by the following code snippet. We randomly crop the image into `224x224` and flip them horizontally for training samples, and use single center crop for validation.

```py
img_size = 224
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if is_training:
  return datasets.ImageFolder(
      train_dir,
      transforms.Compose([
          transforms.RandomResizedCrop(img_size),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))
else:
  return datasets.ImageFolder(
      val_dir,
      transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(img_size),
          transforms.ToTensor(),
          normalize,
      ]))
```

### Transfer a model

Download an ImageNet pre-trained model from TorchVision and fit its weights to the target dataset.

```shell
python transfer.py \
  -a vgg16 \
  -c PATH/TO/CHECKPOINT \
  -d cub200 \
  --dataset-dir PATH/TO/CUB-200/DATASET \
  --epochs 60 \
  --lr 1e-4 \
  --train-batch 32 \
  --wd 1e-4 \
  --pretrained \
  --gpu-id 0
```

Above is the recipe to transfer an ImageNet pre-trained VGG-16 model to the CUB-200 dataset using the same hyperparameters as the paper: 60 epochs, 1e-4 fixed learning rate. 32 batch size and 1e-4 weight decay are deduced from the setting of the fine-tuning after pruning part of the paper.

The models transferred are listed as follows:

| Model  | Dataset | Top-1 Acc. (%) | Download                                                                     |
| ------ | ------- | -------------- | ---------------------------------------------------------------------------- |
| VGG-16 | CUB-200 | 76.355         | [link](https://s3.amazonaws.com/taylor-pruning/vgg16_cub200_transfer.tar.gz) |

### Prune a model

Run the pruning loop proposed by the paper.

```shell
python prune.py -a vgg16 -d cub200 --dataset-dir $DATASET --resume checkpoints/vgg16 -c checkpoints/pruning/vgg16 --epochs 30 --lr 1e-4 --wd 1e-4 --num-prune-iters 10 --num-channels-per-prune 100
```

This command runs 10 pruning iterations, and in each iteration, it prunes 100 channels of activations.

Results will be saved under the checkpoint directory provided.

### Development and Troubleshooting

Please ask any problem you have through issues.
