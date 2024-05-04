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
        'feature_extractor': args.feature_extractor
    }

    kwargs['occupancymodel'] = args.occupancymodel
    kwargs['pretrained'] = args.pretrained
    kwargs['head'] = args.head
    kwargs['sparse_unet'] = args.sparse_unet
    kwargs['buildinginput']  = args.buildinginput
    kwargs['biasinit'] = args.biasinit
    kwargs['sentinelbuildings'] = args.sentinelbuildings
    kwargs['builtuploss'] = args.builtuploss
    kwargs['basicmethod'] = args.basicmethod
    kwargs['twoheadmethod'] = args.twoheadmethod
    return kwargs
