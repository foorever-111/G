# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class TFA(TwoStageDetector):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._link_prototype_bank()

    def _link_prototype_bank(self):
        bank = getattr(getattr(self.roi_head, 'bbox_head', None),
                       'prototype_bank', None)
        if hasattr(self.rpn_head, 'set_prototype_source') and bank is not None:
            self.rpn_head.set_prototype_source(bank)
