import os
import torch
import random
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
import math

from postprocess import Hypothesis
from preprocess import EOS_TKN

class ASR(nn.Module):
    def __init__(self, output_dim: int, encoder_state_size: int,
        decoder_state_size: int, mlp_out_size: int, feature_dim: int,
        tf_rate: float):

        '''
        Input arguments:
        * output_dim (int): The output dimensionality of character predictions
        * encoder_state_size (int): The state size of the encoder LSTMs
        * decoder_state_size (int): The state size of the decoder LSTMs
        * mlp_out_size (int): The output dimensionality of the linear networks
        used in the attention module
        * feature_dim (int): The feature dimensionality of inputs
        * tf_rate (float): The teacher forcing rate in range (0, 1). Higher means
        more teacher forcing.
        '''
        super(ASR, self).__init__()

        enc_out_dim = encoder_state_size * 2

        self.encoder = Listener(encoder_state_size, feature_dim)

        self.attention = Attention(mlp_out_size, enc_out_dim,
            decoder_state_size)

        self.decoder = Speller(decoder_state_size, enc_out_dim)

        self.embed = nn.Embedding(output_dim, decoder_state_size)
        char_dim = output_dim
        self.char_trans = nn.Linear(decoder_state_size, char_dim)

        self.tf_rate = tf_rate

        # In terms of loading a model, this is ok since we load
        # the trained weights after doing this initalization
        self.init_parameters()

    def forward(self, audio_feature, decode_step, teacher=None,
        state_len=None):
        '''
        * audio_feature (Tensor): A [batch_size, seq, feature] fbank
        * decode_step (int): The length of the longest target in the batch
        * teacher (Tensor): A [batch_size, seq] tensor, containing the text
        targets of the batch
        * state_len (list): The length of unpadded fbanks in the input.
        '''

        # encode_feature shape : [batch_size, ~seq/8, listener.state_size*2]
        encode_feature, encode_len = self.encoder(audio_feature, state_len)

        # Attention based decoding
        if teacher is not None:
            teacher = self.embed(teacher)

        # Init (init char = <SOS>, reset all rnn state and cell)
        self.decoder.init_rnn(encode_feature.shape[0], encode_feature.device)
        self.attention.reset_enc_mem()
        batch_size = audio_feature.shape[0]
        last_char = self.embed(torch.zeros((batch_size),
            dtype=torch.long).to(next(self.decoder.parameters()).device))
        output_char_seq = []
        output_att_seq = []

        # Decode
        for t in range(decode_step):
            # Attend (inputs current state of first layer, encoded features)
            # shapes: attn_score [batch_size, encode_steps]
            attention_score, context = self.attention(
                self.decoder.state_list[0], encode_feature, encode_len)
            # Spell (inputs context + embedded last character)
            decoder_input = torch.cat([last_char, context],dim=-1)
            dec_out = self.decoder(decoder_input)

            # To char
            cur_char = self.char_trans(dec_out)

            # Teacher forcing
            if teacher is not None:
                # and t < decode_step - 1:
                if random.random() <= self.tf_rate:
                    last_char = teacher[:,t+1,:]
                else:
                    sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                    last_char = self.embed(sampled_char)
            else:
                last_char = self.embed(torch.argmax(cur_char,dim=-1))

            output_char_seq.append(cur_char)
            output_att_seq.append(attention_score.cpu().detach())

        att_output = torch.stack(output_char_seq, dim=1)

        # shape: [batch_size, encode_steps, decode_steps]
        [batch_size, encode_step, _] = encode_feature.shape
        output_att_seq = torch.stack(output_att_seq, dim=1)
        return encode_len, att_output, output_att_seq

    def decode(self, x, x_len, rnn_lm, mapper, lm_weight):
        '''
        This is only used for inference and testing. Both the ASR
        and the supplied LM should be in validation mode.

        Input arguments:
        x (tensor): A single audio sample, shape = [1, seq, features]
        x_len (list): The ouput of dataset.prepare_x(x)
        rnn_lm (nn.Module): A language model
        mapper (object): A dataset.mapper instance
        '''
        curr_device = next(self.decoder.parameters()).device

        assert len(x.shape) == 3 and x.shape[0] == 1
        bs = x.shape[0]
        decoding_steps = 0
        max_decoding_steps = 200
        # encode_feature shape : [batch_size, ~seq/8, listener.state_size*2]
        encode_feature, encode_len = self.encoder(x, x_len)

        # Init (init char = <SOS>, reset all rnn state and cell)
        self.decoder.init_rnn(bs, curr_device)
        (h1, h2) = rnn_lm.init_hidden(bs, curr_device)
        self.attention.reset_enc_mem()
        batch_size = x.shape[0]
        last_char = self.embed(torch.zeros((batch_size), dtype=torch.long).to(curr_device))
        last_char_idx = torch.LongTensor([0]).to(curr_device)
        output_char_seq = []
        output_att_seq = []
        char_predicts = []
        # we keep decoding until <EOS> has been emitted
        while decoding_steps < max_decoding_steps:
            # Attend (inputs current state of first layer, encoded features)
            # shapes: attn_score [batch_size, encode_steps]
            attention_score, context = self.attention(
                self.decoder.state_list[0], encode_feature, encode_len)
            # Spell (inputs context + embedded last character)
            decoder_input = torch.cat([last_char, context],dim=-1)
            dec_out = self.decoder(decoder_input)

            # To char
            asr_predict = F.log_softmax(self.char_trans(dec_out), dim=-1)
            lm_out, (h1, h2) = rnn_lm(last_char_idx, h1, h2)
            lm_predict = F.log_softmax(lm_out.squeeze(1), dim=-1)
            final_predict = asr_predict + lm_weight * lm_predict
            asr_char = torch.argmax(asr_predict,dim=-1)
            lm_char = torch.argmax(lm_predict,dim=-1)
            predicted_char = torch.argmax(final_predict,dim=-1)

            last_char_idx = torch.LongTensor([predicted_char]).to(curr_device)
            last_char = self.embed(predicted_char)

            output_char_seq.append(final_predict)
            output_att_seq.append(attention_score.cpu().detach())

            if predicted_char.item() == mapper.char_to_ind(EOS_TKN):
                #print("Stopping because of EOS")
                break

            char_predicts.append(mapper.ind_to_char(predicted_char.item()))
            decoding_steps += 1
        return ''.join(c for c in char_predicts)

    def init_parameters(self):
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 3:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        self.embed.weight.data.normal_(0, 1)
        set_forget_bias_to_one(self.decoder.layer_1.bias_ih)
        set_forget_bias_to_one(self.decoder.layer_2.bias_ih)

class Listener(nn.Module):
    def __init__(self, state_size, feature_dim):
        '''
        Input arguments:
        * state_size (int): The state size of pBLSTM units
        * feature_dim (int): The feature dimensionality of the
        input to the listener.
        '''
        super(Listener, self).__init__()

        self.state_size = state_size
        self.out_dim = 2 * self.state_size

        # [batch, seq, features] -> [batch, seq/8, state_size*2]
        '''
        -> [b, s, f] -> lstm_1 -> [b, s, ss] -> down_sample -> [b, s/2, ss*2]
        -> lstm_2 -> [b, s/2, ss] -> down_sample -> [b, s/4, ss*2]
        -> lstm_3 -> [b, s/2, ss] -> down_sample -> [b, s/8, ss*2] <- i.e. state_size*4
        (add one more LSTM at end to get state_size*2)
        '''
        self.blstm_1 = pBLSTM(feature_dim, self.state_size)
        self.blstm_2 = pBLSTM(self.state_size*2*2, self.state_size)
        self.blstm_3 = pBLSTM(self.state_size*2*2, self.state_size)
        self.blstm_4 = nn.LSTM(self.state_size*2*2, self.state_size,
            bidirectional=True)

    def get_outdim(self):
        return self.out_dim

    def forward(self, x, state_len, pack_input=True):
        '''
        Input arguments:
        * x (Tensor): A [batch_size, sequence, features] shaped tensor
        containing input feature banks
        * state_len: A list of unpadded lengths of each sample in the
        batch. This is used for packing the sequences.
        * pack_input (bool): If true, the input to the RNNs will be packed

        Output:
        * x (Tensor): A [batch_size, seq/8, state_size*2] shaped tensor
        * state_len (list): The unpadded lengths of each feature
        '''
        x, _, state_len = self.blstm_1(x, state_len=state_len,
            pack_input=pack_input)
        x, _, state_len = self.blstm_2(x, state_len=state_len,
            pack_input=pack_input)
        x, _, state_len = self.blstm_3(x, state_len=state_len,
            pack_input=pack_input)
        x, _, = self.blstm_4(x)

        return x, state_len

# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, state_size, encoder_out_size):
        '''
        Input arguments:
        * state_size (int): The state size of both LSTM layers
        * encoder_out_size (int): Equal to Listener.out_dim. Used
        to calculate the input dimensionality of the first LSTM layer
        '''
        super(Speller, self).__init__()

        self.layer_1 = nn.LSTMCell(
            input_size=encoder_out_size+state_size,
            hidden_size=state_size)

        self.layer_2 = nn.LSTMCell(
            input_size=state_size,
            hidden_size=state_size)

        self.state_list = []
        self.cell_list = []

        self.state_size = state_size
        self.num_layers = 2

    def init_rnn(self, batch_size, device):
        '''
        Input arguments:
        * batch_size (int): The batch size
        * device (Torch.device): The device which should
        contain the state and cell lists of the Speller
        '''
        self.state_list = [torch.zeros(batch_size,
            self.state_size).to(device)]*self.num_layers
        self.cell_list = [torch.zeros(batch_size,
            self.state_size).to(device)]*self.num_layers

    @property
    def hidden_state(self):
        return [s.clone().detach().cpu() for s in self.state_list], \
            [c.clone().detach().cpu() for c in self.cell_list]

    @hidden_state.setter
    def hidden_state(self, state): # state is a tuple of two
        device = self.state_list[0].device
        self.state_list = [s.to(device) for s in state[0]]
        self.cell_list = [c.to(device) for c in state[1]]

    def forward(self, input_context):
        '''
        Input arguments:
        * input_context (Tensor): A [bs, self.state_size+Listener.out_dim]
        shaped tensor representing one 'step' of input to the speller.
        '''
        self.state_list[0],self.cell_list[0] = self.layer_1(
            input_context,(self.state_list[0],self.cell_list[0]))

        self.state_list[1], self.cell_list[1] = self.layer_2(
            self.state_list[0], (self.state_list[1], self.cell_list[1]))

        return self.state_list[-1]

class Attention(nn.Module):
    def __init__(self, mlp_out_size, encoder_out_size, decoder_state_size):
        super(Attention,self).__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.phi = nn.Linear(decoder_state_size, mlp_out_size, bias=False)
        self.psi = nn.Linear(encoder_out_size, mlp_out_size)

        self.comp_listener_feature = None

    def reset_enc_mem(self):
        self.comp_listener_feature = None
        self.state_mask = None

    def forward(self, decoder_state, listener_feature, state_len):
        '''
        Input arguments:
        * decoder_state (Tensor): A tensor, containing the
        "last" output of the listener
        * listener_feature (Tensor): A batch of shape
        [bs, ~seq/8, listener.out_dim], containing the output
        from the listener
        * state_len (list like): contains the unpadded length
        of each fbank in the current batch. In the case of the text
        auto encoder, these should be the lengths of each unpadded noised
        string in the batch.

        Returns:
        * attention_score (Tensor): which is
        softmax( phi(decoder_state).T @ psi(listener_feature))
        * context (Tensor): which is attention_score @ listener_feature
        '''

        # Store enc state to save time
        if self.comp_listener_feature is None:
            # Maskout attention score for padded states
            # NOTE: mask MUST have all input > 0

            '''
            The state mask might look like this, given the state_len
            of [4, 6, 2]
            | 1 1 1 1 0 0 0 |
            | 1 1 1 1 1 1 0 |
            | 1 1 0 0 0 0 0 |
            '''
            self.state_mask = np.zeros((listener_feature.shape[0],
                listener_feature.shape[1]))
            for idx,sl in enumerate(state_len):
                self.state_mask[idx,sl:] = 1
            self.state_mask = torch.from_numpy(self.state_mask) \
                .type(torch.ByteTensor) \
                .to(decoder_state.device)
            self.comp_listener_feature =  torch.tanh(self.psi(listener_feature))

        comp_decoder_state =  torch.tanh(self.phi(decoder_state))

        energy = torch.bmm(self.comp_listener_feature,
            comp_decoder_state.unsqueeze(2)).squeeze(dim=2)
        energy.masked_fill_(self.state_mask, -float("Inf"))
        attention_score = self.softmax(energy)
        context = torch.bmm(attention_score.unsqueeze(1),
            listener_feature).squeeze(1)

        return attention_score, context

class pBLSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        Input arguments:
        * in_dim (int): The dimensionality of the input
        * out_dim (int): The dimensionality of the output
        '''

        super(pBLSTM, self).__init__()
        self.layer = nn.LSTM(in_dim, out_dim, bidirectional=True,
            batch_first=True)

    def forward(self, input_x, state=None, state_len=None,
        pack_input=False):

        # Forward RNN
        if pack_input:
            assert state_len is not None, \
                "Please specify seq len for pack_padded_sequence."
            input_x = pack_padded_sequence(input_x, state_len, batch_first=True)
        output, hidden = self.layer(input_x, state)

        if pack_input:
            output, state_len = pad_packed_sequence(output,batch_first=True)
            state_len = state_len.tolist()

        # Perform Downsampling
        output = self.downsample(output)

        if state_len is not None:
            state_len=[int(s/2) for s in state_len]
            return output, hidden, state_len

        return output, hidden

    def downsample(self, x):
        '''
        Downsamples the time-axis by a factor of 2
        of the input x.
        Args:
            x: a [batch, seq, feature] tensor
        Out:
            x: a [batch, seq/2, feature*2] tensor.
        (Note: If t is not divisible by two, the last frame
        will be dropped)
        '''
        t_dim = x.shape[1]
        f_dim = x.shape[2]

        if t_dim % 2 != 0:
            # drop the last frame if odd number of frames
            x = x[:, :t_dim-1, :]
            t_dim -= 1

        # concat consecutive frames
        x = x.contiguous().view([-1, int(t_dim/2), f_dim*2])
        return x


if __name__ == '__main__':
    l = Listener(256, 40)
    x = torch.randn(32, 8, 40)
    print(l(x, [8 for _ in range(32)])[0].shape)