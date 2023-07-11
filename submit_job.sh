#!/bin/bash

#SBATCH --time=23:59:00
#SBATCH --ntasks=32
#SBATCH --output euleroutputs/outfile_%J.%I.txt
#SBATCH --mem-per-cpu=1250
#SBATCH --gpus=1
#SBATCH --gres=gpumem:21g
#### ## #SBATCH --mail-type=END,FAIL

#SBATCH -J CyCADA # The job name

source PopMapEnv/bin/activate

# load modules
#module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1 opencv/3.4.3
#cuda/11.6.2
module load eth_proxy 


# python run_train.py --no-osm --satmode --num_epochs 100 --lam_builtmask 0.0 --batch_size 384 --lr_step 10 -lr 1e-4 --lr_step 15 --lr_gamma 0.75 \
#     --model JacobsUNet --feature_extractor vgg11 --loss l1_loss --lam 1.0 --lam_adv 100.0 -S2 --feature_dim 16 --random_season --target_regions pri2017 --adversarial \
#     --classifier v5 --excludeZH --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults

# python run_train.py --no-osm --satmode --num_epochs 100 --lam_builtmask 0.0 --batch_size 384 --lr_step 10 -lr 1e-4 --lr_step 15 --lr_gamma 0.75 \
#     --model JacobsUNet --feature_extractor vgg11 --loss l1_loss --lam 1.0 --lam_adv 0.01 -S2 --feature_dim 16 --random_season --target_regions pri2017 --excludeZH \
#     --adversarial --classifier v7 --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults


python run_train.py -o -s -e 200 -b 32 -lr 1e-5 --model BoostUNet -l log_l1_aug_loss --merge_aug 2 -la 1.0 -dw 2 -dw2 4 -S2 -S1 -f 16 -rse -treg rwa -exZH --val 1 \
    --lr_step 5 --lr_gamma 0.75 --full_aug --num_workers 6 --seed 1612 -gc 0.001 \
    --CyCADA \
    --CyCADASourcecheckpoint /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults/So2Sat/experiment_905_626/best_model.pth \
    --CyCADAGANcheckpoint eu2rwa_cycleganFreeze2 --lam_selfsupervised_consistency 1.0 --lam_targetconsistency 0.1 --lambda_consistency_fake_B 1.0 --lambda_consistency_real_B 100.0 --lambda_popB 100.0 --CyCADAnetG resnet_6blocks \
    --save-dir /cluster/work/igp_psr/metzgern/HAC/code/PopMapResults