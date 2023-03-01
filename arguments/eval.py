import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=8, help='Scaling factor')
parser.add_argument('--in-memory', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--no_params', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--feature-extractor', type=str, default='UResNet', help='Feature extractor for edge potentials')

parser.add_argument('--N', type=int, default=4000, help='N rgb iterations')
parser.add_argument('--Npre', type=int, default=1000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=16, help='N learned iterations with gradients')
parser.add_argument('--pixtransform', default=False, action='store_true', help='eval the pix transformer.')
parser.add_argument('--sdfilter', default=False, action='store_true', help='eval the pix transformer.')
parser.add_argument('--gfilter', default=False, action='store_true', help='eval the pix transformer.')
parser.add_argument('--bicubic', default=False, action='store_true', help='eval the pix transformer.')
