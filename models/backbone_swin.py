# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from torch import nn
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from models.Swin.swin_transformer import SwinTransformer
from models.Swin.config import base_cascade,tiny_cascade,tiny_maskrcnn,small_cascade,small_maskrcnn


Swin_config = dict(base_cascade=base_cascade, tiny_cascade=tiny_cascade,
                   tiny_maskrcnn=tiny_maskrcnn, small_cascade=small_cascade, small_maskrcnn=small_maskrcnn)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)   # tensor_list [mask, tensor]
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone_swin(args):
    position_embedding = build_position_encoding(args)
    model_cfg = Swin_config[args.swin_model]
    backbone = SwinTransformer(embed_dim=model_cfg['embed_dim'],
                               depths=model_cfg['depths'],
                               num_heads=model_cfg['num_heads'],
                               window_size=model_cfg['window_size'],
                               ape=model_cfg['ape'],
                               drop_path_rate=model_cfg['drop_path_rate'],
                               patch_norm=model_cfg['patch_norm'],
                               use_checkpoint=model_cfg['use_checkpoint'],
                               output_dim=model_cfg['output_dim'])
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.outputd_dim
    return model
