

# from model.pomelo import JacobsUNet, PomeloUNet, ResBlocks, UResBlocks, ResBlocksDeep, ResBlocksSqueeze
# from model.ownmodels import BoostUNet
# from model.resblock_pomelo import ResBlocksPomelo

from model.module_pomelo import POMELO_module


model_dict = {
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
        'input_channels': args.Sentinel1 * 2 + args.NIR * 1 + args.Sentinel2 * 3,
        'feature_dim': args.feature_dim,
        'feature_extractor': args.feature_extractor
    }

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
        kwargs['sparse_unet'] = args.sparse_unet
        kwargs['buildinginput']  = args.buildinginput
        kwargs['biasinit'] = args.biasinit
        kwargs['sentinelbuildings'] = args.sentinelbuildings
    return kwargs
