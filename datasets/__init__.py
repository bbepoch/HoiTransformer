# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .hico import build as build_hico
from .vcoco import build as build_vcoco


def build_dataset(image_set, args, test_scale=-1):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file == 'hico':
        return build_hico(image_set, test_scale)
    elif args.dataset_file == 'vcoco':
        return build_vcoco(image_set, test_scale)
    else:
        pass
