#!/bin/bash

#SBATCH -J train_dual_mgn
#SBATCH --partition=dgx2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mail-type=end
#SBATCH --mail-user=chenyongbiao0319@sina.com
#SBATCH -o ./out/source_out/%j.out
#SBATCH -e ./out/source_out/%j.err

module load miniconda3
source activate tenv
python source_training.py
