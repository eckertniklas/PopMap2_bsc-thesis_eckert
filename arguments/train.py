import configargparse

parser = configargparse.ArgumentParser(description='Training Population Estimation')

parser.add_argument('-r', '--resume', type=str, help='if argument is given, script will continue training given model\  ; argument should be name of the model to be trained')
parser.add_argument('-re', '--resume_extractor', type=str, help=' ')
parser.add_argument("-treg", "--target_regions", nargs='+', default=["pri2017"], help="the target domains")
parser.add_argument("-tregtrain", "--target_regions_train", nargs='+', default=["pri2017"], help="the target domains")
parser.add_argument("-S1", "--Sentinel1", action='store_true', help="")
parser.add_argument("-S2", "--Sentinel2", action='store_true', help="")
parser.add_argument("-NIR", "--NIR", action='store_true', help="")
parser.add_argument("-VIIRS", "--VIIRS", action='store_true', help="")
parser.add_argument('-wb', '--weak_batch_size', help='', type=int, default=2)
parser.add_argument('-wvb', '--weak_val_batch_size', help='', type=int, default=1)
# parser.add_argument('-f', '--feature_dim', help='', type=int, default=32)
parser.add_argument('-pret', '--pretrained', help='', action='store_true')
parser.add_argument("-m", "--model", help='', type=str, default="JacobsUNet")
parser.add_argument("-binit", "--biasinit", help='', type=float, default=0.75)
parser.add_argument("-sunet", "--sparse_unet", help='',action='store_true')
parser.add_argument("-omo", "--occupancymodel", help='', action='store_true')
parser.add_argument("-binp", "--buildinginput", help='', action='store_true')
parser.add_argument("-sinp", "--segmentationinput", help='', action='store_true')
parser.add_argument("-senbuilds", "--sentinelbuildings", help='', action='store_true')
parser.add_argument("-uaf", "--useallfeatures", action='store_true', help="")
parser.add_argument('-fe', '--feature_extractor', type=str, help=' ', default="DDA")

# Training
parser.add_argument('-e', '--num_epochs', help='', type=int, default=20)
parser.add_argument('-lr', '--learning_rate', help='', type=float, default=1e-4)
parser.add_argument('-optim', '--optimizer', type=str, default="Adam", choices=["Adam", "SGD"])
parser.add_argument('-gacc', '--gradientaccumulation', action='store_true', help="")
parser.add_argument('-l', '--loss', nargs='+', default=["log_l1_loss"], help="list composed of 'l1_loss', 'log_l1_loss', 'mse_loss', 'log_mse_loss', 'focal_loss','tversky_loss', 'gaussian_nll', 'laplacian_nll', 'L1reg'")
parser.add_argument('-sreg', '--scale_regularization', default=0.01, type=float, help="")
parser.add_argument('-la', '--lam', nargs='+', type=float, default=[1.0], help="list composed of loss weightings")
parser.add_argument("-lw", "--lam_weak", help='', type=float, default=100.0)
parser.add_argument("-lim1", "--limit1", type=int, default=15000000, help="")
parser.add_argument("-lim2", "--limit2", type=int, default=18000000, help="")
parser.add_argument("-lim3", "--limit3", type=int, default=22000000, help="")

parser.add_argument("-head", "--head", default="v3", help="")

parser.add_argument('-wd', '--weightdecay', help='', type=float, default=0.0)
parser.add_argument('-wdu', '--weightdecay_unet', help='', type=float, default=0.0)
parser.add_argument("-rse", "--random_season", action='store_true', help="")
parser.add_argument('-lrs', '--lr_step', help='', type=int, default=5)
parser.add_argument('-lrg', '--lr_gamma', help='', type=float, default=0.75)
parser.add_argument('-gc', '--gradient_clip', help='', type=float, default=0.01)
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')
parser.add_argument('-fa', '--full_aug', action='store_true', help='')
parser.add_argument('-ascAug', '--ascAug', action='store_true', help='')

# misc
parser.add_argument('--save-dir', default='/scratch2/metzgern/HAC/POMELOv2_results', help='Path to directory where models and logs should be saved')
parser.add_argument('-w', '--num_workers', help='', type=int, default=6)
parser.add_argument("-wp", "--wandb_project", help='', type=str, default="POMELOv2")
parser.add_argument('-lt', '--logstep_train', help='', type=int, default=25)
parser.add_argument('-val', '--val_every_n_epochs', help='', type=int, default=1)
parser.add_argument('-wv', '--weak_validation', help='', action='store_true')
parser.add_argument('-testi', '--test_every_i_steps', help='', type=int, default=500000)
parser.add_argument('-vi', '--val_every_i_steps', help='', type=int, default=500000)
parser.add_argument("--seed", help='', type=int, default=1610)
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('-ms', '--max_samples', help='', type=int, default=1e15)
parser.add_argument('-mws', '--max_weak_samples', help='', type=int, default=None)
parser.add_argument('-mwp', '--max_weak_pix', help='', type=int, default=10000000)
parser.add_argument('-mpb', '--max_pix_box', help='', type=int, default=12000000)
parser.add_argument('-tlevel', '--train_level', nargs='+', default=["coarse"] )

args = parser.parse_args()