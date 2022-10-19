'''
Author: Yunxiang Liu u7191378@anu.edu.au
Date: 2022-09-12 21:28:35
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2022-10-19 16:17:38
FilePath: \HoiTransformer\models\__init__.py
Description: build_model
'''
# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .hoitr import build as build_hoitr
from .hoitr import build_II as build_sim_hoitr


def build_model(args):
    return build_hoitr(args)

def build_model_II(args):
    return build_sim_hoitr(args)
