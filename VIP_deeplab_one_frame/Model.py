from seg_decoder import FinalSemanticDecoder
from ins_decoder import FinalInstanceDecoder
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.semantic_depth = FinalSemanticDecoder()
        self.instance = FinalInstanceDecoder()

    def forward(self, inputs):
        semantic_pre, depth_estimation = self.semantic_depth(inputs)
        center_prediction, center_regression = self.instance(inputs)
        return semantic_pre, depth_estimation, center_prediction, center_regression
