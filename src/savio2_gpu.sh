#!/bin/bash
# Job name:
#SBATCH --job-name=daniil
#
# Account:
#SBATCH --account=fc_deepmusic
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=00:10:00
#
## Command(s) to run (example):
module unload python/3.7
module load ml/tensorflow/2.3.0-py37 libsndfile
python train.py

