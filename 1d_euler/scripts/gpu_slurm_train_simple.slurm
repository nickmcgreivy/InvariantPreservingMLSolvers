#!/bin/bash
#SBATCH --job-name=1d_euler_complex_ghost
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-corea
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mcgreivy@princeton.edu

#
# Hyperparameters



module load anaconda3/2021.11 cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0 nvhpc/21.5
conda activate jax-gpu

python 1D_Euler_Simple_Initial_Conditions.py
