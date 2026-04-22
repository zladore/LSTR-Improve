# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn

from rekognition_online_action_detection.utils.registry import Registry

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
}


@FEATURE_HEADS.register('THUMOS')
@FEATURE_HEADS.register('TVSeries')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        self.with_visual = False
        self.with_motion = False
        self.with_text = False

        if cfg.INPUT.MODALITY == 'visual':
            self.with_visual = True
        elif cfg.INPUT.MODALITY == 'motion':
            self.with_motion = True
        elif cfg.INPUT.MODALITY == 'twostream':
            self.with_visual = True
            self.with_motion = True
        elif cfg.INPUT.MODALITY == 'threestream':
            self.with_visual = True
            self.with_motion = True
            self.with_text = True
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        fusion_size = 0
        if self.with_visual:
            fusion_size += FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        if self.with_motion:
            fusion_size += FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
        if self.with_text:
            if cfg.INPUT.TEXT_FEATURE_DIM <= 0:
                raise RuntimeError('cfg.INPUT.TEXT_FEATURE_DIM must be > 0 for threestream')
            fusion_size += cfg.INPUT.TEXT_FEATURE_DIM

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            self.input_linear = nn.Sequential(
                nn.Linear(fusion_size, self.d_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_linear = nn.Identity()

    def forward(self, visual_input=None, motion_input=None, text_input=None):
        fusion_inputs = []

        if self.with_visual:
            if visual_input is None:
                raise RuntimeError('visual_input is required for current modality')
            fusion_inputs.append(visual_input)

        if self.with_motion:
            if motion_input is None:
                raise RuntimeError('motion_input is required for current modality')
            fusion_inputs.append(motion_input)

        if self.with_text:
            if text_input is None:
                raise RuntimeError('text_input is required for current modality')
            fusion_inputs.append(text_input)

        if len(fusion_inputs) == 1:
            fusion_input = fusion_inputs[0]
        else:
            fusion_input = torch.cat(fusion_inputs, dim=-1)

        fusion_input = self.input_linear(fusion_input)
        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)
