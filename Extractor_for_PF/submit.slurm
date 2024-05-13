#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --mem=185G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chirayu.gupta@gmail.com
#SBATCH --job-name="timAhcal_valid"

module load iiser/apps/cuda/11.4
module load python/3.9.8
python3 makepickles.py
