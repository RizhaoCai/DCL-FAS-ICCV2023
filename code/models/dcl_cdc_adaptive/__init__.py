"""
    bc contain many networks with different architecture. But all these networks are for binary classification and use cross encropy as loss for optimization
"""
from models.bc_ewc_cdc_adaptive.trainer import Trainer
from models.bc_ewc_cdc_adaptive.custom_config import _C as custom_cfg
import torch


__all__ = [
    'Trainer',
    'custom_cfg'
]

