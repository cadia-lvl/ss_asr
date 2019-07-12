import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class TextAutoEncoder(nn.Module):
    def __init__(self, char_dim:int, emb_dim:int=128,
            state_size:int=256, num_layers:int=2):
        '''
        Input arguments:
        * char_dim (int): Number of symbols used in the ASR
        * emb_dim (int): The dimensionality of the character 
        embeddings
        * state_size (int): The state size of the bLSTM layers.

        * an encoder: The text encoder takes one-hot character 
        encodings as input. It is composed of a single embedding 
        layer, to convert these inputs to character vectors, 
        followed by two bLSTM layers.

        * a decoder: The decoder is shared with the 
        LAS model, such that training this autoencoder 
        also trains the parameters of the LAS decoder.
        '''
        super(TextAutoEncoder, self).__init__()

        self.encoder = TextEncoder(char_dim, emb_dim, state_size, num_layers)

    def forward(self, asr, y, y_noised, decode_step, noise_lens=None):
        '''
        Input arguments:
        * asr (nn.Module): An instance of the src.ASR class
        * y (Tensor): A Tensor of shape [batch, seq]
        where y[i, j] is the encoded version of the j-th character
        from the i-th sample in the y batch.
        * y_noised (Tensor): A [batch_size, =< seq] tensor, containing the 
        noisy text. 
        * decode_step (int): The length of the longest target in the batch.
        * noise_lens (List like): All unpadded target lengths (after noising)
        * asr (nn.Module): 
        y and y_noised are the same, but characters may have been dropped from
        y_noised.
        
        To run this forward without any noise, supply y_noised=y and 
        noise_lens=y_lens
        
        '''

        '''First, text autoencoder specifics only'''
        y_encoded = self.encoder(y_noised) # [bs, seq, encoder.state_size*2]

        '''Then, we do attendAndSpell from the LAS ASR'''
        asr.decoder.init_rnn(y_encoded.shape[0], y_noised.device)
        asr.attention.reset_enc_mem()

        # y shape: [bs, seq, dataset.get_char_dim()]
        y = asr.embed(y)
    
        batch_size = y_noised.shape[0]
        last_char = asr.embed(torch.zeros((batch_size),
            dtype=torch.long).to(next(asr.decoder.parameters()).device))
        output_char_seq = []
        output_att_seq = []

        # Decode (We decode as many steps as there are at maximum, unpadded wise, in
        # the non-noisy batch)
        for t in range(decode_step):
            # Attend (inputs current state of first layer, encoded features)
            attention_score, context = asr.attention(
                asr.decoder.state_list[0], y_encoded, noise_lens)
            # Spell (inputs context + embedded last character)                
            decoder_input = torch.cat([last_char, context],dim=-1)
            dec_out = asr.decoder(decoder_input)
            
            # To char
            cur_char = asr.char_trans(dec_out)

            # Teacher forcing
            if (y is not None) and t < decode_step - 1:
                if random.random() <= asr.tf_rate:
                    last_char = y[:,t+1,:]
                else:
                    sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                    last_char = asr.embed(sampled_char)
            else:
                last_char = asr.embed(torch.argmax(cur_char,dim=-1))

            output_char_seq.append(cur_char)

        att_output = torch.stack(output_char_seq, dim=1)

        return noise_lens, att_output

class TextEncoder(nn.Module):
    def __init__(self, char_dim, emb_dim, state_size, num_layers):
        super(TextEncoder, self).__init__()
        
        self.emb = nn.Embedding(char_dim, emb_dim)
        self.blstm = nn.LSTM(input_size=emb_dim, hidden_size=state_size,
            num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, y):
        emb = self.emb(y)
        # y_encoded shape : [batch, seq, self.blstm.state_size*2]
        y_encoded, _ = self.blstm(emb)
        return y_encoded