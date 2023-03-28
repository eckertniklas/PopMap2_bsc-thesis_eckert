#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --ntasks=4
#SBATCH --output euleroutputs/outfile_%J.%I.txt
#SBATCH --mem-per-cpu=10000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:21g
#SBATCH --mail-type=END,FAIL

#SBATCH -J {#1} # The job name

source PopMapEnv/bin/activate

# load modules
#module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1 opencv/3.4.3
#cuda/11.6.2
module load eth_proxy 

# python run_train.py --no-osm --satmode --num_epochs 100 --lam_builtmask 0.0 --batch_size 384 --lr_step 10 -lr 3e-4 --lr_step 15 --lr_gamma 0.75 \
#     --model JacobsUNet --feature_extractor vgg11 --loss l1_loss --lam 1000 --lam_adv 100.0 -S2 --feature_dim 16 --random_season --target_region pri2017 --adversarial \
#     --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults --head v1 --classifier v2

python run_train.py --no-osm --satmode --num_epochs 100 --lam_builtmask 0.0 --batch_size 128 --lr_step 10 -lr 3e-4 --lr_step 15 --lr_gamma 0.75 \
    --model JacobsUNet --feature_extractor vgg11 --loss l1_loss --lam 1000 --lam_adv 100.0 -S2 --feature_dim 16 --random_season --target_region pri2017 --adversarial \
    --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults --classifier v5 --in_memory