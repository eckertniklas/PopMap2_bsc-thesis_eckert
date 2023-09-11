

from model.pomelo import JacobsUNet, PomeloUNet, ResBlocks, UResBlocks, ResBlocksDeep, ResBlocksSqueeze
from model.ownmodels import BoostUNet
from model.resblock_pomelo import ResBlocksPomelo

from model.module_pomelo import POMELO_module


model_dict = {
    "JacobsUNet": JacobsUNet,
    "PomeloUNet": PomeloUNet,
    "ResBlocks": ResBlocks,
    "ResBlocksSqueeze": ResBlocksSqueeze,
    "UResBlocks": UResBlocks,
    "ResBlocksDeep": ResBlocksDeep,
    "BoostUNet": BoostUNet,
    "ResBlocksPomelo": ResBlocksPomelo,
    "POMELO_module": POMELO_module
}




def get_model_kwargs(args, model_name):
    """
    :param args: arguments
    :param model_name: name of the model
    :return: kwargs for the model
    """

    # kwargs for the model
    kwargs = {
        'input_channels': args.Sentinel1 * 2 + args.NIR * 1 + args.Sentinel2 * 3 + args.VIIRS * 1,
        # 'input_channels': args.Sentinel1 * 2 + args.NIR * 1 + args.Sentinel2 * 3 + args.VIIRS * 1 + args.buildinginput * 1 + args.segmentationinput * 1,
        'feature_dim': args.feature_dim,
        'feature_extractor': args.feature_extractor
    }

    # additional kwargs for the Jacob's model
    if model_name == 'JacobsUNet':
        kwargs['classifier'] = args.classifier if args.adversarial else None
        kwargs['head'] = args.head
        kwargs['down'] = args.down
        kwargs['occupancymodel'] = args.occupancymodel
        kwargs['pretrained'] = args.pretrained
        kwargs['dilation'] = args.dilation
        kwargs['replace7x7'] = args.replace7x7
    if model_name == 'BoostUNet':
        # assert args.Sentinel1
        # assert args.Sentinel2
        kwargs['classifier'] = args.classifier if args.adversarial else None
        kwargs['down'] = args.down
        kwargs['down2'] = args.down2
        kwargs['occupancymodel'] = args.occupancymodel 
        kwargs['useallfeatures'] = args.useallfeatures
        kwargs['pretrained'] = args.pretrained
        kwargs['dilation'] = args.dilation
        kwargs['replace7x7'] = args.replace7x7
    if model_name == 'ResBlockPomelo':
        kwargs['classifier'] = args.classifier if args.adversarial else None
        kwargs['occupancymodel'] = args.occupancymodel
    if model_name == 'POMELO_module': 
        kwargs['occupancymodel'] = args.occupancymodel
        kwargs['pretrained'] = args.pretrained
        kwargs['dilation'] = args.dilation
        kwargs['replace7x7'] = args.replace7x7
        kwargs['parent'] = args.parent
        kwargs['down'] = args.down
        kwargs['experiment_folder'] = args.experiment_folder
        kwargs['useposembedding'] = args.useposembedding
        kwargs['head'] = args.head
        kwargs['grouped'] = args.grouped
        kwargs['lempty_eps'] = args.lempty_eps
        kwargs['dropout'] = args.dropout
    return kwargs
