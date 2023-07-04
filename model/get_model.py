

from model.pomelo import JacobsUNet, PomeloUNet, ResBlocks, UResBlocks, ResBlocksDeep, ResBlocksSqueeze
from model.ownmodels import BoostUNet


model_dict = {
    "JacobsUNet": JacobsUNet,
    "PomeloUNet": PomeloUNet,
    "ResBlocks": ResBlocks,
    "ResBlocksSqueeze": ResBlocksSqueeze,
    "UResBlocks": UResBlocks,
    "ResBlocksDeep": ResBlocksDeep,
    "BoostUNet": BoostUNet,
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
        'feature_dim': args.feature_dim,
        'feature_extractor': args.feature_extractor
    }

    # additional kwargs for the Jacob's model
    if model_name == 'JacobsUNet':
        kwargs['classifier'] = args.classifier if args.adversarial else None
        kwargs['head'] = args.head
        kwargs['down'] = args.down
    if model_name == 'BoostUNet':
        assert args.Sentinel1
        assert args.Sentinel2
        kwargs['classifier'] = args.classifier if args.adversarial else None
        # kwargs['head'] = args.head
        kwargs['down'] = args.down
        kwargs['down2'] = args.down2
        kwargs['occupancymodel'] = args.occupancymodel 
        kwargs['useallfeatures'] = args.useallfeatures
    return kwargs
