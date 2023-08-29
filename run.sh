#!/bin/bash
#SBATCH --job-name=xu
#SBATCH --account=Project_2002243
#SBATCH --partition=gpusmall
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1,nvme:180
$SCRATCH

module load pytorch/1.10

srun nvidia-smi

# Please set the dataset address in configs.py

python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 1
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget EuroSAT --n_shot 1
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget ISIC --n_shot 1
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget ChestX --n_shot 1
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget EuroSAT --n_shot 5
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget ISIC --n_shot 5
#python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget ChestX --n_shot 5