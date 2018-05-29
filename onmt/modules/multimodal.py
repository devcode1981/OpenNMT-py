import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt
import onmt.io
from onmt.Utils import aeq

class GeneratorGate(nn.Module):
    pass

class MultiModalGenerator(nn.Module):
    def __init__(self, old_generator):
        super(MultiModalGenerator, self).__init__()
        self.linear = old_generator[0]
        self.gate = GeneratorGate(self.linear.weight.size(1))
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, img_feats):
        pass

class MultiModalLossCompute(onmt.Loss.LossComputeBase):
    def __init__(self, generator, tgt_vocab, label_smoothing=0.0)
        super(MultiModalLossCompute, self).__init__(
            generator, tgt_vocab)

    def _compute_loss(self, FIXME):
        pass
