#!/bin/bash
#SBATCH --job-name=fourier_modulate_normalize_32
#SBATCH --output=fourier_modulate_normalize_32.txt
#SBATCH --mem 128GB
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH -p owners
#SBATCH --gres gpu:1
singularity exec --nv --bind /home:/home /scratch/users/chenkaim/singularity_images/surrogate-latest_torch.simg \
            python3 train_model.py
