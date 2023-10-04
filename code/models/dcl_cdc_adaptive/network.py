import timm
import torchvision.models as models
from torch import nn
from .timm_vit import _create_vit_adapter
import logging
from timm.models.vision_transformer import vit_base_patch16_224
from .convpass import set_Convpass



def build_net(config):
    """

    :param config:
    :return:
    """
    imagetnet_pretrain = config.MODEL.IMAGENET_PRETRAIN
    model_arch = config.MODEL.ARCH
    fix_backbone = config.MODEL.FIX_BACKBONE
    num_classes = config.MODEL.NUM_CLASSES # Default 2
    fix_head = False

    adaptive_type = config.MODEL.ADAPTIVE_TYPE

    model = vit_base_patch16_224(imagetnet_pretrain, num_classes=num_classes)
    set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=False, conv_type='cdc', adaptive_type=adaptive_type)


    for name, p in model.named_parameters():
        if fix_backbone and not fix_head:
            if 'adapter' in name or 'head' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        if fix_backbone and fix_head:
            if 'adapter' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    return model

if __name__ == '__main__':
    pass

