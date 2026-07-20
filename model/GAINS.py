import torch
import torch
import torch.nn as nn


SOS_ID = 0
EOS_ID = 0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import info

SOS_ID = -1
EOS_ID = -1


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1,
                                                                                                        self.output_dim)

        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length  # total length to decode
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.sos_id = vocab_size - 1
        self.eos_id = vocab_size - 1
        self.gpu = gpu

    def forward(self, x, encoder_hidden=None, encoder_outputs=None):

        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:  # if not given x, then we are inferring!
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden,
                                                    encoder_outputs,
                                                    self.gpu)  # hidden is layer-wise out, out is final output
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def decode(step_, step_output_, step_attn_):
            decoder_outputs.append(step_output_)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn_)
            # if step_ % 2 == 0:  # sample index, should be in [1, index-1]
            #     index = step_ // 2 % 10 // 2 + 3
            #     symbols_ = decoder_outputs[-1][:, 1:index].topk(1)[1] + 1
            # else:  # sample operation, should be in [7, 11]
            #     symbols_ = decoder_outputs[-1][:, 7:].topk(1)[1] + 7
            symbols_ = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols_)

            eos_batches = symbols_.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step_) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols_

        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden

    def _validate_args(self, x, encoder_hidden, encoder_outputs, gpu=0):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)

        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda(gpu)
            max_length = self.length
        else:
            max_length = x.size(1)

        return x, batch_size, max_length

    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self.forward(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, x: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        pass


class RNNDecoder(Decoder):

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu
                 ):
        super(RNNDecoder, self).__init__(
            layers,
            vocab_size,
            hidden_size,
            dropout,
            length, gpu)

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.init_input = None
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)  # class num + 1 eos

    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)  # attention from decoder_output and encoder_output
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn


def construct_decoder(fe, args) -> Decoder:
    name = args.method_name
    size = fe.ds_size
    info(f'Construct Decoder with method {name}...')
    if name == 'rnn':
        return RNNDecoder(
            layers=args.decoder_layers,
            vocab_size=size + 1,
            hidden_size=args.decoder_hidden_size,
            dropout=args.decoder_dropout,
            length=size,
            gpu=args.gpu
        )
    elif name == 'transformer':
        assert False
    else:
        assert False

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logger import info


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size):
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, seq_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_seq_emb = torch.mean(new_encoder_outputs, dim=1)
        new_seq_emb = F.normalize(new_seq_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, new_encoder_outputs, new_seq_emb

    def forward(self, x):
        pass


class RNNEncoder(Encoder):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout
                 ):
        super(RNNEncoder, self).__init__(layers, vocab_size, hidden_size)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # batch x length x hidden_size
        embedded = self.dropout(embedded)

        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out

        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value


def construct_encoder(fe, args) -> Encoder:
    name = args.method_name
    size = fe.ds_size
    info(f'Construct Encoder with method {name}...')
    if name == 'rnn':
        return RNNEncoder(
            layers=args.encoder_layers,
            vocab_size=size + 1,
            hidden_size=args.encoder_hidden_size,
            dropout=args.encoder_dropout,
            mlp_layers=args.mlp_layers,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_dropout=args.encoder_dropout
        )
    elif name == 'transformer':
        assert False
    else:
        assert False



# gradient based automatic feature selection
class GAINS(nn.Module):
    def __init__(self,
                 fe,
                 args
                 ):
        super(GAINS, self).__init__()
        self.style = args.method_name
        self.gpu = args.gpu
        self.encoder = construct_encoder(fe, args)
        self.decoder = construct_decoder(fe, args)
        if self.style == 'rnn':
            self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, feat_emb, predict_value = self.encoder.forward(input_variable)
        decoder_hidden = (feat_emb.unsqueeze(0), feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        feat = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, feat

    def generate_new_feature(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
            self.encoder.infer(input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_feat_emb.unsqueeze(0), new_feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(None, new_encoder_hidden, new_encoder_outputs)
        new_feat_seq = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_feat_seq
