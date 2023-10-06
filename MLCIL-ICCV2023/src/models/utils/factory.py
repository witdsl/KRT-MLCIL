import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL

def my_create_model(args, num_classes):
    """Create a model, with model_name and num_classes
    """
    model_params = {'args': args, 'num_classes': num_classes}
    model_name = args.model_name.lower()

    if model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    
    else:
        print("model: {} not found !!".format(model_name))
        exit(-1)

    return model