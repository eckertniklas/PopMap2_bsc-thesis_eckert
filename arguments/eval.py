import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)
parser.add_argument('-r', '--resume', type=str, help='if argument is given, skript will continue training '
                                                            'given model\
                                                            ; argument should be name of the model to be trained')
parser.add_argument("-treg", "--target_regions", nargs='+', default=["pri2017"], help="the target domains")
parser.add_argument("-S1", "--Sentinel1", action='store_true', help="")
parser.add_argument("-S2", "--Sentinel2", action='store_true', help="")
parser.add_argument("-NIR", "--NIR", action='store_true', help="")
parser.add_argument("-VIIRS", "--VIIRS", action='store_true', help="")
parser.add_argument('-f', '--feature_dim', help='', type=int, default=32)
parser.add_argument("-m", "--model", help='', type=str, default="JacobsUNet")
parser.add_argument("-omo", "--occupancymodel", help='', action='store_true')
parser.add_argument("-sinp", "--segmentationinput", help='', action='store_true')
parser.add_argument("-binp", "--buildinginput", help='', action='store_true')
parser.add_argument("-senbuilds", "--sentinelbuildings", help='', action='store_true')
parser.add_argument('-leeps', '--lempty_eps', default=0.0, type=float, help="")
parser.add_argument("-uaf", "--useallfeatures", action='store_true', help="")
parser.add_argument("-fs", "--fourseasons", action='store_true', help="")
parser.add_argument("-dw", "--down", help='', type=int, default=2)
parser.add_argument("-dw2", "--down2", help='', type=int, default=2)
parser.add_argument('-fe', '--feature_extractor', type=str, help=' ', default="resnet18")
parser.add_argument('-pret', '--pretrained', help='', action='store_true')
parser.add_argument("-r77", "--replace7x7", action='store_true', help="")
parser.add_argument("-posemb", "--useposembedding", help='', action='store_true')
parser.add_argument("-dil", "--dilation", help='', type=int, default=1)
parser.add_argument('-par', '--parent', type=str, help=' ', default=None)
parser.add_argument("-gr", "--grouped", help='', action='store_true')
parser.add_argument('-tlevel', '--train_level', nargs='+', default=["fine"] )
parser.add_argument('-eeps', '--empty_eps', default=0.0, type=float, help="")
parser.add_argument("-dro", "--dropout", help='', type=float, default=0.0)
parser.add_argument("-sunet", "--sparse_unet", help='',action='store_true')

#Training
parser.add_argument("-clasif", "--classifier", default="v8", help="")
parser.add_argument("-head", "--head", default="v1", help="")
parser.add_argument("-adv", "--adversarial", action='store_true', help="")

# misc
parser.add_argument('--save-dir', default='/scratch2/metzgern/HAC/POMELOv2_results', help='Path to directory where models and logs should be saved')
parser.add_argument('-w', '--num_workers', help='', type=int, default=6)
parser.add_argument("-wp", "--wandb_project", help='', type=str, default="POMELOv2")
parser.add_argument("--seed", help='', type=int, default=1610)
parser.add_argument("--in_memory", action='store_true', help='')

args = parser.parse_args()