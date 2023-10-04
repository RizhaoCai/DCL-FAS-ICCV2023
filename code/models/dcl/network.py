import timm
import torchvision.models as models
from torch import nn
from .timm_vit import _create_vit_adapter
import logging
from timm.models.vision_transformer import vit_base_patch16_224
from .convpass import set_Convpass


def get_model_from_torchvision(arch_name, imagetnet_pretrain, num_classes):
    """
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)
    googlenet = models.googlenet(pretrained=True)
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
    mnasnet = models.mnasnet1_0(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)
    """
    net = models.__dict__[arch_name](pretrained=imagetnet_pretrain)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_classes)
    return net


def get_ghost_net():
    from models.bc.ghost_net import ghostnet
    return ghostnet(num_classes=2)

def build_net(config):
    """

    :param config:
    :return:
    """
    imagetnet_pretrain = config.MODEL.IMAGENET_PRETRAIN
    model_arch = config.MODEL.ARCH
    fix_backbone = config.MODEL.FIX_BACKBONE
    num_classes = config.MODEL.NUM_CLASSES # Default 2
    fix_head = config.MODEL.FIX_HEAD # Whether to fix the classification head ()
    cdc_theta = config.MODEL.CDC_THETA

    if 'net' in model_arch.lower() and model_arch.lower() in models.__dict__.keys():
        model =  get_model_from_torchvision(model_arch, imagetnet_pretrain, num_classes)
        if fix_backbone:
            logging.info('Fix Backbone')
            for name, p in model.named_parameters():
                # import pdb; pdb.set_trace()
                if fix_head:
                    if 'layer4' in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                elif not fix_head:
                    if 'fc' in name or 'layer4' in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False



        return model
    elif 'timm' in model_arch:
        backbone_name = model_arch.split('-')[-1]
        # for example: backbone_name='vit_base_patch16_224'
        model = timm.create_model(backbone_name, pretrained=imagetnet_pretrain, num_classes=num_classes).cuda()

        if fix_backbone:
            for name, p in model.named_parameters():
                if not fix_head:
                    if 'head' in name or 'blocks.11' in name or name == 'norm.bias' or name == 'norm.weight':
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

                if fix_head:
                    if 'blocks.11' in name or name == 'norm.bias' or name == 'norm.weight':
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        return model

    elif 'adapter' in model_arch:
        backbone_name = model_arch.split('-')[-1]
        def _vit_base_patch16_224(pretrained=True, **kwargs):
            """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
            ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
            """

            model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
            model = _create_vit_adapter('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
            return model


        if backbone_name == 'vit_base_patch16_224':
            model = _vit_base_patch16_224(imagetnet_pretrain, num_classes=num_classes)
        for name, p in model.named_parameters():
            # import pdb; pdb.set_trace()
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

    elif 'convpass' in model_arch:
        # convpass-conv-vit_base_patch16_224
        backbone_name = model_arch.split('-')[-1]
        # conv_type: conv, cdc, cdc_matrix_5x5shared, cdc_matrix_4x4unshared, cdc_matrix_5x5unshared
        conv_type = model_arch.split('-')[-2]
        if backbone_name == 'vit_base_patch16_224':
            model = vit_base_patch16_224(imagetnet_pretrain, num_classes=num_classes)  # todo

        if 'cdc' in conv_type:
            set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=False, conv_type=conv_type, cdc_theta=cdc_theta)
        elif conv_type == 'conv':
            set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=True, conv_type=conv_type )

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

