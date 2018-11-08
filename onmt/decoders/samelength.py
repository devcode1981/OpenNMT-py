""" Base Class and function for Decoders """

from __future__ import division
import torch
import torch.nn as nn

import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory
from onmt.decoders.decoder import RNNDecoderBase


class SameLengthDecoder(RNNDecoderBase):
    """
    Recurrent attention-less decoder,
    which is synchronized to the encoder,
    and can thus only produce a string of the same length as the input.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, encoder_size, dropout=0.0, embeddings=None):
        self.encoder_size = encoder_size
        super(SameLengthDecoder, self).__init__(
            rnn_type, bidirectional_encoder, num_layers,
            hidden_size, dropout=dropout, embeddings=embeddings)

        # Remove the standard attention.
        del self.attn

    def init_state(self, src, memory_bank,
                   encoder_final=None, with_cache=False):
        """ Init decoder state with zeros. """
        batch_size = memory_bank.size(1)
        state_size = (self.num_layers, batch_size, self.hidden_size)
        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = (memory_bank.data.new(*state_size).zero_(),
                                    memory_bank.data.new(*state_size).zero_())
        else:  # GRU
            self.state["hidden"] = (memory_bank.data.new(*state_size).zero_(),)

    def forward(self, tgt, memory_bank, memory_lengths=None,
                step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # memory_bank padded to same length as emb
        memory_bank = torch.cat(
            [memory_bank, memory_bank.data.new(1, memory_bank.size(1), memory_bank.size(2)).zero_()])
        # If single-stepping, extract the right timestep from memory_bank
        if step is not None:
            memory_bank_slice = memory_bank[step:step+1]
        else:
            memory_bank_slice = memory_bank
        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank_slice, memory_lengths=memory_lengths)

        # Update the state with the result.
        output = dec_outs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        self.update_state(dec_state, output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

        dummy_att_size = [memory_bank.size(0), memory_bank.size(1), memory_bank.size(1)]
        dummy_attns = {'std': memory_bank.data.new(*dummy_att_size).zero_()}
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, dummy_attns

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the
                          encoder RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict): empty
        """
        assert not self._copy  # will not be supported
        assert not self._coverage  # will not be supported

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # concatenate embedding and encoder output
        inp = torch.cat([emb, memory_bank], dim=2)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, dec_state = self.rnn(inp, self.state["hidden"][0])
        else:
            rnn_output, dec_state = self.rnn(inp, self.state["hidden"])

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        dec_outs = rnn_output  # .transpose(0, 1).contiguous()
        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size + self.encoder_size
