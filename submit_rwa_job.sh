#!/bin/bash

#SBATCH --time=23:59:00
#SBATCH --ntasks=12
#SBATCH --output euleroutputs/outfile_%J.%I.txt
#### ## #SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23g
#### ## #SBATCH --mail-type=END,FAIL


#SBATCH -J GB # The job name

nvidia-smi

# source PopMapEnv2/bin/activate
# source /cluster/home/neckert/venv/PopMapEnv/bin/activate

source /cluster/work/igp_psr/metzgern/HAC/code/repocode/PopMap2/PopMapEnv/bin/activate

# load modules
# module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1 opencv/3.4.3
# module load cuda/11.8.0
module load eth_proxy #gdal/3.4.3

# python run_train.py -o -s -e 200 -b 16 -lr 1e-4 --model POMELO_module -l log_l1_loss L1reg -la 1.0 -dw 2 -dw2 2 -S2 -S1 -NIR -f 8 -rse -treg rwa -tregtrain rwa -exZH --val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 200 -b 16 -lr 1e-4 --model POMELO_module -l log_l1_loss -la 1.0 -dw 2 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH --val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults


# python run_train.py -o -s -e 200 -b 16 -lr 1e-3 --model POMELO_module -l log_l1_loss -la 1.0 -dw 1 -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH --val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --useposembedding --weightdecay_pos 1e-4 --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults


# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v2 -l log_l1_loss -la 1.0 --scale_regularization 10.0 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.0 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.825 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.001 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.0 -dw 1 -S1 -f 8 -rse -treg rwa -tregtrain uga --weak_validation -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --weightdecay_pos 0.0 --sentinelbuildings --pretrained --gradientaccumulation \
#      --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.01 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.0 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.0 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 0.0 --scale_regularizationL2 10.0 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py -o -s -e 100 -b 16 -lr 1e-3 --model POMELO_module --head v3 -l log_l1_loss -la 1.0 --scale_regularization 10.0 -dw 1 -S2 -NIR -S1 -f 8 -rse -treg rwa -tregtrain rwa -exZH -val 1 --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 \
#     --seed 1612 -gc 0.01 --supmode weaksup --lam_weak 100 --nomain --val_every_n_epochs 1 --weak_batch_size 1 --occupancymodel --buildinginput --useposembedding --weightdecay_pos 0.0 --sentinelbuildings -r77 --pretrained --gradientaccumulation \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

python run_train.py -S2 -NIR -S1 -treg rwa -tregtrain rwa --seed 1600 -occmodel -wd 0.0000001 -senbuilds -pret --biasinit 0.9407 --save-dir /cluster/work/igp_psr/neckert/results --builtuploss --buildinginput --segmentationinput --lambda_builtuploss 1.0 --twoheadmethod

# python run_train.py -S2 -NIR -S1 -treg rwa -tregtrain rwa --seed 1600 -occmodel -wd 0.0000001 -senbuilds -pret --biasinit 0.9407 --save-dir /cluster/work/igp_psr/neckert/results --builtuploss --buildinginput --segmentationinput --lambda_builtuploss 1.0 --basicmethod

# python run_train.py -S2 -NIR -S1 -treg rwa -tregtrain rwa --seed 1600 -occmodel -wd 0.0000001 -senbuilds -pret --biasinit 0.9407 --save-dir /cluster/work/igp_psr/neckert/results

# python run_train.py -S2 -NIR -S1 -treg pricp2 -tregtrain pricp2 --seed 1600 -occmodel -wd 0.0000005 -senbuilds -pret --biasinit 0.4119 --save-dir /cluster/work/igp_psr/neckert/results
