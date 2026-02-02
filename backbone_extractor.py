import torch
import torchvision.models as models
from torchvision.models.vgg import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights, VGG11_BN_Weights, VGG13_BN_Weights, VGG16_BN_Weights, VGG19_BN_Weights
from torchvision.models.densenet import DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights

def get_backbone(model_name):

    if model_name == 'vgg11':
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg13':
        model = models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg19':
        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet161':
        model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet169':
        model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet201':
        model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
    else:
        raise Exception('Model not implemented')
    
    backbone = model.features

    return _preprocess(backbone, model_name)

def _preprocess(model, model_name):

    model_name = model_name.upper()

    if model_name.startswith('VGG'):
        first_conv = model[0] # Layer to update
    elif model_name.startswith('DENSE'):
        first_conv = model.conv0
    else:
        raise Exception('Model not implemented')
    
    old_weights = first_conv.weight
    new_weights = torch.mean(old_weights, dim=1, keepdim=True)

    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size, 
        stride=first_conv.stride, 
        padding=first_conv.padding)
    
    new_conv.weight.data = new_weights
    
    if model_name.startswith('VGG'):
        model[0] = new_conv
    elif model_name.startswith('DENSE'):
        model.conv0 = new_conv
    else:
        raise Exception('Model not implemented')
    
    return model   