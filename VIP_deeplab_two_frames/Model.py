from seg_decoder import FinalSemanticDecoder
from ins_decoder import FinalInstanceDecoder
from next_frame_decoder import InstanceNextDecoder
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.semantic_depth = FinalSemanticDecoder()
        self.instance = FinalInstanceDecoder()
        self.instance_next = InstanceNextDecoder()

    def forward(self, inputs, inputs_next):
        semantic_pre, depth_estimation = self.semantic_depth(inputs)
        center_prediction, center_regression, _ = self.instance(inputs)
        center_regression_next = self.instance_next(inputs_next, inputs)

        return semantic_pre, depth_estimation, center_prediction, center_regression, center_regression_next
