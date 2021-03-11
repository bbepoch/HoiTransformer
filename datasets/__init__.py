# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .hico import build as build_hico


def build_dataset(image_set, args, test_scale=-1):
    if args.dataset_file == 'hico':
        return build_hico(image_set, test_scale)
    raise ValueError(f'dataset {args.dataset_file} not supported')
