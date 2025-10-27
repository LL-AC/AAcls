import timm
from .simpleViT import SimpleViT
from .resnet import resnet
from .AACls import AACls

def build_model(args):
    
    model_name = args.model.lower()
    
    if model_name == 'your model':
        model = resnet()
    else:
        raise NotImplementedError(f"{model_name} is not implementde")
    
    return model