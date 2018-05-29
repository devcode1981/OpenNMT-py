import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt
import onmt.io
from onmt.Utils import aeq

class GeneratorGate(nn.Module):
    def __init__(self, img_feat_size, vocab_size):
        super(GeneratorGate, self).__init__()
        self.gate = nn.Linear(img_feat_size, vocab_size, bias=True)

    def forward(self, inp, img_feats):
        print('inp', inp.shape)
        print('img_feats', img_feats.shape)
        print('gated', self.gate(img_feats).shape)
        return inp * F.sigmoid(self.gate(img_feats))


class MultiModalGenerator(nn.Module):
    def __init__(self, old_generator, img_feat_size):
        super(MultiModalGenerator, self).__init__()
        self.linear = old_generator[0]
        print('old_generator', self.linear.weight.shape)
        self.gate = GeneratorGate(
            img_feat_size,
            self.linear.weight.size(0))
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, img_feats):
        proj = self.linear(hidden)
        gated = self.gate(proj, img_feats)
        return self.logsoftmax(gated)
        pass

class MultiModalLossCompute(onmt.Loss.NMTLossCompute):
    def _compute_loss(self, batch, output, target, img_feats):
        img_gate = FIXME # compute gate from img_feats. Replicate for timesteps.
        scores = self.generator(self._bottle(output), img_gate)

        ### Copypasta from superclass
        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats
        ### Copypasta ends
