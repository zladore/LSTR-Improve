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

        self.fusion_mode = getattr(cfg.MODEL.FEATURE_HEAD, 'FUSION_MODE', 'concat')
        self.text_gate_bias_init = float(
            getattr(cfg.MODEL.FEATURE_HEAD, 'TEXT_GATE_BIAS_INIT', -2.0)
        )

        if self.with_text and self.fusion_mode == 'gated_residual_text':
            if not self.with_visual or not self.with_motion:
                raise RuntimeError('gated_residual_text currently requires visual+motion+text')
            if not cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
                raise RuntimeError('gated_residual_text requires LINEAR_ENABLED=True')
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES == -1:
                raise RuntimeError('gated_residual_text requires LINEAR_OUT_FEATURES > 0')
            if cfg.INPUT.TEXT_FEATURE_DIM <= 0:
                raise RuntimeError('TEXT_FEATURE_DIM must be > 0')

            vm_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE] + FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES

            self.vm_linear = nn.Sequential(
                nn.Linear(vm_size, self.d_model),
                nn.ReLU(inplace=True),
            )
            self.text_linear = nn.Linear(cfg.INPUT.TEXT_FEATURE_DIM, self.d_model)
            self.gate_linear = nn.Linear(self.d_model * 2, self.d_model)

            nn.init.constant_(self.gate_linear.bias, self.text_gate_bias_init)

            # concat mode modules unused here
            self.input_linear = None

        else:
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
        if self.with_text and self.fusion_mode == 'gated_residual_text':
            if visual_input is None:
                raise RuntimeError('visual_input is required')
            if motion_input is None:
                raise RuntimeError('motion_input is required')
            if text_input is None:
                raise RuntimeError('text_input is required')

            vm_input = torch.cat([visual_input, motion_input], dim=-1)
            base_feat = self.vm_linear(vm_input)              # 主干：RGB+Flow
            text_feat = self.text_linear(text_input)          # 文本分支
            gate = torch.sigmoid(
                self.gate_linear(torch.cat([base_feat, text_feat], dim=-1))
            )
            fusion_input = base_feat + gate * text_feat
            return fusion_input

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
