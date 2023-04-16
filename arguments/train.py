import configargparse
import numpy as np

parser = configargparse.ArgumentParser()
# general

parser = configargparse.ArgumentParser(description='Training Population Estimation')

parser.add_argument('-o', '--no-osm', help='If set, Training Skript wont use OSM Data', default=False,
                    action='store_true')
parser.add_argument('-s', '--satmode', help='Using Satellite Data only', default=False,
                    action='store_true')
parser.add_argument('-r', '--resume', type=str, help='if argument is given, skript will continue training '
                                                            'given model\
                                                            ; argument should be name of the model to be trained')

# parser.add_argument("-d", "--dataset", type=str, default="So2Sat", help="the source domain") 
parser.add_argument("-treg", "--target_regions", nargs='+', default=["pri2017"], help="the target domains")
parser.add_argument("-S1", "--Sentinel1", action='store_true', help="")
parser.add_argument("-S2", "--Sentinel2", action='store_true', help="")
parser.add_argument("-NIR", "--NIR", action='store_true', help="")
parser.add_argument("-VIIRS", "--VIIRS", action='store_true', help="")
parser.add_argument('-b', '--batch_size', help='', type=int, default=64)
parser.add_argument('-f', '--feature_dim', help='', type=int, default=32)
parser.add_argument("-m", "--model", help='', type=str, default="JacobsUNet")
parser.add_argument("-dw", "--down", help='', type=int, default=2)
parser.add_argument('-fe', '--feature_extractor', type=str, help=' ', default="resnet18")
parser.add_argument('-smo', '--supmode', type=str, help="'unsup', 'weaksup'", default="unsup")

#Training
parser.add_argument('-e', '--num_epochs', help='', type=int, default=20)
parser.add_argument('-lr', '--learning_rate', help='', type=float, default=1e-4)
parser.add_argument("-exZH", "--excludeZH", action='store_true', help="")
parser.add_argument('-l', '--loss', nargs='+', default=["l1_loss"], help="list composed of 'l1_loss', 'log_l1_loss', 'mse_loss', 'log_mse_loss', 'focal_loss','tversky_loss")
parser.add_argument('-la', '--lam', nargs='+', type=float, default=[1.0], help="list composed of loss weightings")
parser.add_argument("-lw", "--lam_weak", help='', type=float, default=1.0)
parser.add_argument("-adv", "--adversarial", action='store_true', help="")
parser.add_argument("-ladv", "--lam_adv", help='', type=float, default=0.1)
parser.add_argument("-cor", "--CORAL", action='store_true', help="")
parser.add_argument("-lc", "--lam_coral", help='', type=float, default=1.0)
parser.add_argument("-mmd", "--MMD", action='store_true', help="")
parser.add_argument("-lmmd", "--lam_mmd", help='', type=float, default=1.0)
parser.add_argument("-clasif", "--classifier", default="v8", help="")
parser.add_argument("-head", "--head", default="v1", help="")

parser.add_argument('-wd', '--weightdecay', help='', type=float, default=0.0)
parser.add_argument('-ls', '--lassoreg', help='Lasso style regularization of the preds.', type=float, default=0.0)
parser.add_argument('-tv', '--tv', help='Total variation regularization of the preds.', type=float, default=0.0)
parser.add_argument("-rse", "--random_season", action='store_true', help="")
parser.add_argument('-lrs', '--lr_step', help='', type=int, default=10)
parser.add_argument('-lrg', '--lr_gamma', help='', type=float, default=0.75)
parser.add_argument('-gc', '--gradient_clip', help='', type=float, default=0.01)
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')
parser.add_argument('-fa', '--full_aug', action='store_true', help='')
parser.add_argument("-lm", "--lam_builtmask", help='', type=float, default=0.)
parser.add_argument("-ld", "--lam_dense", help='', type=float, default=1.)

# misc
parser.add_argument('--save-dir', default='/scratch2/metzgern/HAC/POMELOv2_results', help='Path to directory where models and logs should be saved')
parser.add_argument('-w', '--num_workers', help='', type=int, default=6)
parser.add_argument("-wp", "--wandb_project", help='', type=str, default="POMELOv2")
parser.add_argument('-lt', '--logstep_train', help='', type=int, default=5)
parser.add_argument('-val', '--val_every_n_epochs', help='', type=int, default=1)
parser.add_argument("--seed", help='', type=int, default=1610)
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('-ms', '--max_samples', help='', type=int, default=1e15)
parser.add_argument('-mws', '--max_weak_samples', help='', type=int, default=None)
parser.add_argument("--in_memory", action='store_true', help='')
parser.add_argument("--merge_aug", action='store_true', help='')
parser.add_argument("-wma", "--weak_merge_aug", action='store_true', help='')

args = parser.parse_args()