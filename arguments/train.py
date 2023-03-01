import configargparse


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

parser.add_argument("-d", "--dataset", help='', type=str, default="So2Sat")
parser.add_argument('-b', '--batch_size', help='', type=int, default=256)
parser.add_argument("-m", "--model", help='', type=str, default="JacobsUNet")
#Training
parser.add_argument('-e', '--num_epochs', help='', type=int, default=20)
parser.add_argument('-lr', '--learning_rate', help='', type=float, default=1e-4)
parser.add_argument('-w', '--num_workers', help='', type=int, default=12)
parser.add_argument('-wd', '--weightdecay', help='', type=float, default=0.0)
parser.add_argument('-lrs', '--lr_step', help='', type=int, default=3)
parser.add_argument('-lrg', '--lr_gamma', help='', type=float, default=1.0)
parser.add_argument('-gc', '--gradient_clip', help='', type=float, default=0.01)
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')

# misc
parser.add_argument('--save-dir', default='/home/pf/pfstaff/projects/metzgern_Sat2Pop/POMELOv2_results', help='Path to directory where models and logs should be saved')
parser.add_argument("-wp", "--wandb_project", help='', type=str, default="POMELOv2")
parser.add_argument('-lt', '--logstep_train', help='', type=int, default=20)
parser.add_argument('-val', '--val_every_n_epochs', help='', type=int, default=1)
parser.add_argument("--seed", help='', type=int, default=1610)
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])

args = parser.parse_args() 