import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable

import onmt
import onmt.io
from onmt.Utils import aeq
from onmt.modules.Transformer import TransformerEncoder


class MultiModalNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, bridge, decoder, multigpu=False, imgw=False):
        self.multigpu = multigpu
        super(MultiModalNMTModel, self).__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder
        self.imgw = imgw

    def forward(self, src, tgt, lengths, dec_state=None, img_feats=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        assert img_feats is not None
        tgt = tgt[:-1]  # exclude last target from inputs

        if self.imgw:
            enc_final, memory_bank = self.encoder(src, img_feats, lengths)
        else:
            enc_final, memory_bank = self.encoder(src, lengths)
        if self.bridge is not None:
            memory_bank = self.bridge(memory_bank, img_feats, src.size(0))
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class MultiModalMemoryBankGate(nn.Module):
    def __init__(self, bank_size, img_feat_size, add=0):
        super(MultiModalMemoryBankGate, self).__init__()
        self.bank_to_gate = nn.Linear(
            bank_size, bank_size, bias=False)
        self.feat_to_gate = nn.Linear(
            img_feat_size, bank_size, bias=True)
        #nn.init.constant_(self.feat_to_gate.bias, 1.0) # newer pytorch
        nn.init.constant(self.feat_to_gate.bias, 1.0)
        self.add = add

    def forward(self, bank, img_feats, n_time):
        feat_to_gate = self.feat_to_gate(img_feats)
        feat_to_gate = feat_to_gate.expand(n_time, -1, -1)
        bank_to_gate = self.bank_to_gate(bank)
        #bank_to_gate = bank_to_gate.view(-1, bank_to_gate.size(2))
        #print('bank_to_gate flat', bank_to_gate.shape)
        gate = F.sigmoid(feat_to_gate + bank_to_gate) + self.add
        gate = gate / (1. + self.add)
        return bank * gate


class MultiModalGenerator(nn.Module):
    def __init__(self, old_generator, img_feat_size, add=0, use_hidden=False):
        super(MultiModalGenerator, self).__init__()
        self.linear = old_generator[0]
        self.vocab_size = self.linear.weight.size(0)
        self.gate = nn.Linear(img_feat_size, self.vocab_size, bias=True)
        #nn.init.constant_(self.gate.bias, 1.0) # newer pytorch
        nn.init.constant(self.gate.bias, 1.0)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.add = add
        self.use_hidden = use_hidden
        if use_hidden:
            self.hidden_to_gate = nn.Linear(
                self.linear.weight.size(1), self.vocab_size, bias=False)

    def forward(self, hidden, img_feats, n_time):
        proj = self.linear(hidden)
        pre_sigmoid = self.gate(img_feats)
        if self.use_hidden:
            pre_sigmoid = pre_sigmoid.repeat(n_time, 1)
            hidden_to_gate = self.hidden_to_gate(hidden)
            gate = F.sigmoid(pre_sigmoid + hidden_to_gate) + self.add
        else:
            gate = F.sigmoid(pre_sigmoid) + self.add
            gate = gate.repeat(n_time, 1)
        gate = gate / (1. + self.add)
        return self.logsoftmax(proj * gate)


class MultiModalLossCompute(onmt.Loss.NMTLossCompute):
    def _compute_loss(self, batch, output, target, img_feats):
        scores = self.generator(self._bottle(output), img_feats, output.size(0))

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


class MultiModalTransformerEncoder(TransformerEncoder):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O



    Args:
       num_layers (int): number of encoder layers
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    """
    def __init__(self, num_layers, hidden_size, img_feat_size,
                 dropout, embeddings):
        super(MultiModalTransformerEncoder, self).__init__(
            num_layers, hidden_size, dropout, embeddings)
        self.img_to_emb = nn.Linear(img_feat_size, hidden_size, bias=True)

    def forward(self, input, img_feats, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        #s_len, n_batch, emb_dim = emb.size()
        img_emb = self.img_to_emb(img_feats).unsqueeze(0)
        # prepend image "word"
        emb = torch.cat([img_emb, emb], dim=0)

        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        # expand mask to account for image "word"
        words = torch.cat([words[:, 0:1], words], dim=1)

        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return Variable(emb.data), out.transpose(0, 1).contiguous()
