#!/bin/bash

#SBATCH --time=23:59:00
#SBATCH --ntasks=8
#SBATCH --output euleroutputs/outfile_%J.%I.txt
# ## #### SBATCH --mem-per-cpu=8000
#SBATCH --mem-per-cpu=11000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#### ## #SBATCH --mail-type=END,FAIL

#SBATCH -J GB # The job name

source PopMapEnv/bin/activate

# load modules
#module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1 opencv/3.4.3
#cuda/11.6.2
module load eth_proxy gdal/3.4.3


python run_train.py -o -s --val_every_i_steps 100 -test_every_i_steps 100 -e 100 -b 16 -lr 1e-4 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.01 -dw 1 -S2 -NIR -S1 -f 8 -rse \
    -treg rwa uga -tregtrain uga --weak_validation -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 --seed 1612 -gc 0.0001 --supmode weaksup --lam_weak 100 --nomain \
    --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --weightdecay_pos 0.0 --sentinelbuildings --pretrained --gradientaccumulation \
    --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults
