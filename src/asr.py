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

class Seq2Seq(nn.Module):
    ''' Seq2Seq model, including Encoder/Decoder(s)'''
    def __init__(self, output_dim, model_para):
        super(Seq2Seq, self).__init__()
        
        self.encoder_state_size = model_para['encoder']['state_size']
        self.decoder_state_size = model_para['decoder']['state_size']
        self.enc_out_dim = model_para['encoder']['state_size'] * 2 * 2

        self.encoder = Listener(self.encoder_state_size, model_para['feature_dim'])

        self.attention = Attention(model_para['attention']['mlp_out_size'],
            self.enc_out_dim, self.decoder_state_size)

        self.decoder = Speller(self.decoder_state_size, self.enc_out_dim)

        self.embed = nn.Embedding(output_dim, self.decoder_state_size)
        self.char_dim = output_dim
        self.char_trans = nn.Linear(self.decoder_state_size,self.char_dim)

        self.tf_rate = model_para['tf_rate']

        self.global_step = 0
        self.best_val_ed = 10.0

        self.init_parameters()

    def set_global_step(self, step:int):
        self.global_step = step

    def get_global_step(self):
        return self.global_step

    def set_best_val(self, val):
        self.best_val_ed = val

    def get_best_val(self):
        return self.best_val_ed

    def load_lm(self,decode_lm_weight,decode_lm_path,**kwargs):
        # Load RNNLM (for inference only)
        self.rnn_lm = torch.load(decode_lm_path)
        self.rnn_lm.eval()
    
    def clear_att(self):
        self.attention.reset_enc_mem()

    def forward(self, audio_feature, decode_step, teacher=None, state_len=None):
        bs = audio_feature.shape[0]
        
        # Encodes
        encode_feature,encode_len = self.encoder(audio_feature,state_len)

        att_output = None
        att_maps = None

        # Attention based decoding
        if teacher is not None:
            teacher = self.embed(teacher)
        
        # Init (init char = <SOS>, reset all rnn state and cell)
        self.decoder.init_rnn(encode_feature)
        self.attention.reset_enc_mem()
        last_char = self.embed(torch.zeros((bs),dtype=torch.long).to(next(self.decoder.parameters()).device))
        output_char_seq = []
        output_att_seq = [[]]
    
        # Decode
        for t in range(decode_step):
            # Attend (inputs current state of first layer, encoded features)
            attention_score,context = self.attention(self.decoder.state_list[0],encode_feature,encode_len)
            # Spell (inputs context + embedded last character)                
            decoder_input = torch.cat([last_char,context],dim=-1)
            dec_out = self.decoder(decoder_input)
            
            # To char
            cur_char = self.char_trans(dec_out)

            # Teacher forcing
            if (teacher is not None):
                if random.random() <= self.tf_rate:
                    last_char = teacher[:,t+1,:]
                else:
                    sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                    last_char = self.embed(sampled_char)
            else:
                last_char = self.embed(torch.argmax(cur_char,dim=-1))

            output_char_seq.append(cur_char)
            
            for head,a in enumerate(attention_score):
                output_att_seq[head].append(a.cpu())

        att_output = torch.stack(output_char_seq,dim=1)
        att_maps = [torch.stack(att,dim=1) for att in output_att_seq]

        return encode_len, att_output, att_maps

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
        set_forget_bias_to_one(self.decoder.lstm.bias_ih)
    
    def beam_decode(self, audio_feature, decode_step, state_len,decode_beam_size):
        '''beam decode returns top N hyps for each input sequence'''
        assert audio_feature.shape[0] == 1
        assert self.training == False
        self.decode_beam_size = decode_beam_size
        
        # Encode
        encode_feature,encode_len = self.encoder(audio_feature,state_len)
        if decode_step==0:
            decode_step = int(encode_len[0])
        
        # Init.
        cur_device = next(self.decoder.parameters()).device
        candidates = None
        att_output = None
        att_maps = None
        lm_hidden = None

        # Store attention map if location-aware
        store_att = self.attention.mode == 'loc'
        
        # Init (init char = <SOS>, reset all rnn state and cell)
        self.decoder.init_rnn(encode_feature)
        self.attention.reset_enc_mem()
        last_char = self.embed(torch.zeros((1),dtype=torch.long).to(cur_device))
        last_char_idx = torch.LongTensor([[0]])
        
        # beam search init
        final_outputs, prev_top_outputs, next_top_outputs = [], [], []
        prev_top_outputs.append(Hypothesis(self.decoder.hidden_state, self.embed, output_seq=[], output_scores=[], 
                                            lm_state=None, att_map = None)) # WIERD BUG here if all args. are not passed...
        # Decode
        for t in range(decode_step):
            for prev_output in prev_top_outputs:
                
                # Attention
                self.decoder.hidden_state = prev_output.decoder_state
                self.attention.prev_att = None if prev_output.att_map is None else prev_output.att_map.to(cur_device)
                attention_score,context = self.attention(self.decoder.state_list[0],encode_feature,encode_len)
                decoder_input = torch.cat([prev_output.last_char,context],dim=-1)
                dec_out = self.decoder(decoder_input)
                cur_char = F.log_softmax(self.char_trans(dec_out), dim=-1)

                # Joint RNN-LM decoding
                if self.decode_lm_weight>0:
                    last_char_idx = prev_output.last_char_idx.to(cur_device)
                    lm_hidden, lm_output = self.rnn_lm(last_char_idx, [1], prev_output.lm_state)
                    cur_char += self.decode_lm_weight * F.log_softmax(lm_output.squeeze(1), dim=-1)

                # Beam search
                topv, topi = cur_char.topk(self.decode_beam_size)
                prev_att_map =  self.attention.prev_att.clone().detach().cpu() if store_att else None 
                final, top = prev_output.addTopk(topi, topv, self.decoder.hidden_state, att_map=prev_att_map,
                                                    lm_state=lm_hidden)
                # Move complete hyps. out
                if final is not None:
                    final_outputs.append(final)
                    if self.decode_beam_size ==1:
                        return final_outputs
                next_top_outputs.extend(top)
            
            # Sort for top N beams
            next_top_outputs.sort(key=lambda o: o.avgScore(), reverse=True)
            prev_top_outputs = next_top_outputs[:self.decode_beam_size]
            next_top_outputs = []
            
            final_outputs += prev_top_outputs
            final_outputs.sort(key=lambda o: o.avgScore(), reverse=True)
        
        return final_outputs[:self.decode_beam_size]


class Listener(nn.Module):
    def __init__(self, state_size, feature_dim):
        super(Listener, self).__init__()
        
        self.state_size = state_size

        # [batch, seq, features] -> [batch, seq, 512] 
        self.blstm_1 = pBLSTM(feature_dim, self.state_size)
        self.blstm_2 = pBLSTM(self.state_size*2*2, self.state_size)
        self.blstm_3 = pBLSTM(self.state_size*2*2, self.state_size)
        self.blstm_4 = pBLSTM(self.state_size*2*2, self.state_size)

    def forward(self, x, enc_len):
        x, _, enc_len = self.blstm_1(x, state_len=enc_len, pack_input=True)
        x, _, enc_len = self.blstm_2(x, state_len=enc_len, pack_input=True)
        x, _, enc_len = self.blstm_3(x, state_len=enc_len, pack_input=True)
        x, _, enc_len = self.blstm_4(x, state_len=enc_len, pack_input=True)

        return x, enc_len

# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, state_size, encoder_out_size):
        super(Speller, self).__init__()
        
        self.lstm = nn.LSTMCell(
            input_size=encoder_out_size+state_size,
            hidden_size=state_size)
        
        self.state_list = []
        self.cell_list = []

        self.state_size = state_size

    def init_rnn(self, context):
        self.state_list = [torch.zeros(context.shape[0],self.state_size).to(context.device)]
        self.cell_list = [torch.zeros(context.shape[0],self.state_size).to(context.device)]

    @property
    def hidden_state(self):
        return [s.clone().detach().cpu() for s in self.state_list], [c.clone().detach().cpu() for c in self.cell_list]

    @hidden_state.setter
    def hidden_state(self, state): # state is a tuple of two list
        device = self.state_list[0].device
        self.state_list = [s.to(device) for s in state[0]]
        self.cell_list = [c.to(device) for c in state[1]]
    
    def forward(self, input_context):
        self.state_list[0],self.cell_list[0] = self.lstm(input_context,(self.state_list[0],self.cell_list[0]))

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
        # Store enc state to save time
        if self.comp_listener_feature is None:
            # Maskout attention score for padded states
            # NOTE: mask MUST have all input > 0 
            self.state_mask = np.zeros((listener_feature.shape[0],listener_feature.shape[1]))
            for idx,sl in enumerate(state_len):
                self.state_mask[idx,sl:] = 1
            self.state_mask = torch.from_numpy(self.state_mask).type(torch.ByteTensor).to(decoder_state.device)
            self.comp_listener_feature =  torch.tanh(self.psi(listener_feature))

        comp_decoder_state =  torch.tanh(self.phi(decoder_state))

        energy = torch.bmm(self.comp_listener_feature, comp_decoder_state.unsqueeze(2)).squeeze(dim=2)
        energy.masked_fill_(self.state_mask,-float("Inf"))
        attention_score = [self.softmax(energy)]
        context = torch.bmm(attention_score[0].unsqueeze(1),listener_feature).squeeze(1)

        return attention_score,context


class pBLSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(pBLSTM, self).__init__()

        self.layer = nn.LSTM(in_dim, out_dim, bidirectional=True, batch_first=True)

    def forward(self, input_x, state=None, state_len=None, pack_input=False):
        # Forward RNN
        if pack_input:
            assert state_len is not None, "Please specify seq len for pack_padded_sequence."
            input_x = pack_padded_sequence(input_x, state_len, batch_first=True)
        output,hidden = self.layer(input_x,state)
        
        if pack_input:
            output, state_len = pad_packed_sequence(output,batch_first=True)
            state_len = state_len.tolist()

        # Perform Downsampling
        output = self.downsample(output)
        if state_len is not None:
            state_len=[int(s/2) for s in state_len]
            return output,hidden,state_len

        return output,hidden
    
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