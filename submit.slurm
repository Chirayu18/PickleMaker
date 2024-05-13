#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks=4
#SBATCH --mem=100g
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mail-user=chirayu.gupta@gmail.com
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name="timAhcal_valid"

folder="/home/chirayugupta/test"
#v2->trueE, no semi
#v1->semi_dscb, loss fun= dscb

#export PYTHONUNBUFFERED=1
#module load ohpc
#module load iiser/apps/cuda/11.4
#module load cmake/3.14.3
#module swap gnu8 cdac/compiler/gcc/10.2.0
#module load python/3.9.8

module load cdac/spack/0.17
source /home/apps/spack/share/spack/setup-env.sh
spack load python@3.8.2
source /home/apps/iiser/pytorch-venv/bin/activate

#module load cdac/compiler/gcc/10.2.0
#module spider graphviz
#source activate torch1.8
/home/chirayugupta/DRN/The_DRN_for_HGCAL/train $folder /home/chirayugupta/pickles/AToGG_pickles_1M_good --nosemi --idx_name all --target trueE --in_layers 3 --mp_layers 4 --out_layers 2  --agg_layers 2 --valid_batch_size 100 --train_batch_size 100  --lr_sched Const --max_lr 0.0001 --pool mean --hidden_dim 128 --n_epochs 100 &>> $folder/training.log
