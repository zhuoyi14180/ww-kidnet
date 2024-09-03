# ww-KidNet Pediatric Brain Tumor Segmentation 

## Training Script

This repository contains the training script for KidNet, a hybrid CNN-Transformer architecture designed for pediatric brain tumor segmentation. The script supports multiple models, including TransBTS, UNet3D, ViT2D, UNet2D, and SwinUNETR. The script leverages PyTorch Distributed Data Parallel (DDP) for efficient multi-GPU training. 



## Requirements

- Python 3.8 or later
- PyTorch 1.7.0 or later
- CUDA 10.2 or later (for GPU training)
- NVIDIA GPUs (A-series GPUs recommended)



## Installation

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-repo/kidnet-brats_ped_2023.git
cd ww-kidnet
```

```
pip install -r requirements.txt
```



## Usage

The training script can be executed with various command-line arguments. Below is a detailed explanation of each available argument.



### Command-Line Arguments

- `--model` (default: `'transbts'`): The model architecture to use for training. Options include:
  - `'transbts'`: Transformer-based 3D model.
  - `'unet3d'`: 3D U-Net model.
  - `'vit2d'`: Vision Transformer in 2D.
  - `'unet2d'`: 2D U-Net model.
  - `'swinunetr'`: Swin Transformer-based U-Net.
- `--dataset` (default: `'brats_ped_2023'`): The dataset to use for training. Options include:
  - `'brats_ped_2023'`: Pediatric brain tumor dataset.
  - `'brats_2019'`: Adult brain tumor dataset.
- `--lr` (default: `0.0002`): Learning rate for the optimizer.
- `--weight_decay` (default: `1e-5`): Weight decay for the optimizer.
- `--amsgrad` (default: `True`): Boolean flag to use AMSGrad variant of the Adam optimizer.
- `--criterion` (default: `'softmax_dice'`): The loss function to use during training. Must match the criterion available in the `criterion` module.
- `--num_class` (default: `4`): The number of classes for segmentation.
- `--seed` (default: `42`): Random seed for reproducibility.
- `--num_workers` (default: `8`): Number of worker threads for data loading.
- `--batch_size` (default: `6`): Batch size for training.
- `--start_epoch` (default: `1`): Starting epoch for training (useful when resuming training).
- `--end_epoch` (default: `400`): Total number of epochs for training.
- `--save_freq` (default: `5000`): Frequency (in iterations) to save the model checkpoint.
- `--resume` (default: `'kidnet-brats_ped_2023-YYYY-MM-DD'`): Name of the checkpoint to resume training from. Automatically includes the current date.
- `--load` (default: `False`): Boolean flag to load from a checkpoint specified in `--resume`.



### Example Commands

#### 1. Train TransBTS on the Pediatric Dataset:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20003 ./train.py --model=transbts --start_epoch=1 --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.001 --seed=1000 --batch_size=8
```

#### 2. Resume Training from a Checkpoint:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20003 ./train.py --model=swinunetr --dataset brats_ped_2023 --resume 'swinunetr-brats_ped_2023-2024-09-03' --load True
```



### Output and Logs

The script will log training progress, including loss and Dice scores for each class, to a log file named with the format `<model>-<dataset>-<date>.txt`. Model checkpoints will be saved periodically in the specified checkpoint directory.



### Final Model and Statistics

Once training is complete, the final model checkpoint and training statistics (e.g., Dice scores) will be saved in the `CHECK_POINT_DIR` and `COLLECTION_DIR` respectively.



## Citation

If you use KidNet or this training script in your research, please consider citing this work.

```
@misc{kidnet2024,
  author = {Your Name},
  title = {KidNet: A Hybrid CNN-Transformer Architecture for Pediatric Brain Tumor Segmentation},
  year = {2024},
  publisher = {Zhuoyi},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zhuoyi14180/ww-kidnet}}
}
```



## Contact

If you would like to obtain the model weight files used in the experiments or all of the source code for KidNet, or any questions or issues, please open an issue on this repository or contact Zhuoyi Zhang at zhuoyi14180@gmail.com.
