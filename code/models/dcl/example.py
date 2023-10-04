import torch

model_path = [

    '/home/rizhao/projects/DCL_FAS/output/vit_adapter/vit_base_patch16_224_Adam_pretrain_update_adapter/HiFi+CelebA+SIW/ft_REPLAY/ckpt/best.ckpt',
    '/home/rizhao/projects/DCL_FAS/output/vit_adapter/vit_base_patch16_224_Adam_pretrain_update_adapter/HiFi+CelebA+SIW/ft_REPLAY/ft_CASIA/ckpt/best.ckpt',
    '/home/rizhao/projects/DCL_FAS/output/vit_adapter/vit_base_patch16_224_Adam_pretrain_update_adapter/HiFi+CelebA+SIW/ft_REPLAY/ft_CASIA/ft_MSU/ft_HKBU/ckpt/best.ckpt'
]

state_dict0 = torch.load(model_path[0], map_location='cpu')
state_dict1 = torch.load(model_path[1], map_location='cpu')
state_dict2 = torch.load(model_path[2], map_location='cpu')

fc_weight=state_dict0['model_state']
fc_weight1=state_dict1['model_state']
fc_weight2=state_dict2['model_state']

import IPython; IPython.embed()
