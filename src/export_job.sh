#!/bin/bash
# Job name:
#SBATCH --job-name=processing
#
# Account:
#SBATCH --account=fc_deepmusic
#
# Partition:
#SBATCH --partition=savio
#
# Request one node:
#SBATCH --nodes=1
#
# Specify number of tasks for use case (example):
#SBATCH --ntasks-per-node=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=24:02:00
#
## Command(s) to run (example):
module load ml/tensorflow/2.5.0-py37 libsndfile
python export_job.sh