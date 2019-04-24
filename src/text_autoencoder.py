import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


import random

class TextAutoEncoder(nn.Module):
    def __init__(self, char_dim:int, emb_dim:int=128,
            state_size:int=256, speller=None, attention=None, 
            char_emb=None, char_trans=None, tf_rate=0.0):
        '''
        Input arguments:
        * char_dim (int): Number of symbols used in the ASR
        * emb_dim (int): The dimensionality of the character 
        embeddings
        * state_size (int): The state size of the bLSTM layers.
        The text autoencoder has three components:
        * a noise model: The noise model drops each character
        with a probability p
        * speller (nn.Module): A src.asr.Speller instance which
        is shared with the main ASR.
        * attention (nn.Module): A src.asr.Attention instance which
        is shared with the main ASR.
        * char_emb (nn.Module): The embedding used in the LAS ASR
        * char_trans (nn.Module): A linear layer, converting from the
        decoder state size to the character dim
        * tf_rate (float): The teacher forcing rate.
        
        * an encoder: The text encoder takes one-hot character 
        encodings as input. It is composed of a single embedding 
        layer, to convert these inputs to character vectors, 
        followed by two bLSTM layers.

        * a decoder: The decoder is shared with the 
        LAS model, such that training this autoencoder 
        also trains the parameters of the LAS decoder.
        '''
        super(TextAutoEncoder, self).__init__()

        self.speller = speller
        self.attention = attention
        self.char_embed = char_emb
        self.char_trans = char_trans

        self.emb = nn.Embedding(char_dim, emb_dim)
        self.blstm = nn.LSTM(input_size=emb_dim, hidden_size=state_size,
            num_layers=2, bidirectional=True, batch_first=True)

        self.tf_rate = tf_rate

    def forward(self, y, y_noised, decode_step, noise_lens=None):
        '''
        Input arguments:
        * y (Tensor): A Tensor of shape [batch, seq]
        where y[i, j] is the encoded version of the j-th character
        from the i-th sample in the y batch.
        * y_noised (Tensor): A [batch_size, =< seq] tensor, containing the 
        noisy text. 
        * decode_step (int): The length of the longest target in the batch.
        * noise_lens (List like): All unpadded target lengths (after noising)
        
        y and y_noised are the same, but characters may have been dropped from
        y_noised.
        '''

        '''First, text autoencoder specifics only'''
        # shape : [batch, seq, emb_dim]
        emb = self.emb(y_noised)
        # y_encoded shape : [batch, seq, self.blstm.state_size*2]
        y_encoded, _ = self.blstm(emb)

        '''Then, we do attendAndSpell from the LAS ASR'''
        self.speller.init_rnn(y_encoded.shape[0], y_noised.device)
        self.attention.reset_enc_mem()

        # y shape: [bs, seq, dataset.get_char_dim()]
        y = self.char_embed(y)
    
        batch_size = y_noised.shape[0]
        last_char = self.char_embed(torch.zeros((batch_size),
            dtype=torch.long).to(next(self.speller.parameters()).device))
        output_char_seq = []
        output_att_seq = []

        # Decode (We decode as many steps as there are at maximum, unpadded wise, in
        # the non-noisy batch)
        for t in range(decode_step):
            # Attend (inputs current state of first layer, encoded features)
            attention_score, context = self.attention(
                self.speller.state_list[0], y_encoded, noise_lens)
            # Spell (inputs context + embedded last character)                
            decoder_input = torch.cat([last_char, context],dim=-1)
            dec_out = self.speller(decoder_input)
            
            # To char
            cur_char = self.char_trans(dec_out)

            # Teacher forcing
            if (y is not None) and t < decode_step - 1:
                if random.random() <= self.tf_rate:
                    last_char = y[:,t+1,:]
                else:
                    sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                    last_char = self.char_embed(sampled_char)
            else:
                last_char = self.char_embed(torch.argmax(cur_char,dim=-1))

            output_char_seq.append(cur_char)

        att_output = torch.stack(output_char_seq, dim=1)

        return noise_lens, att_output

def test_solver():
    '''
    The text autoencoder is trained with the same end-to-end maximum 
    likelihood objective as the speech recognition model, with the same sampling
    procedure
    '''
    from dataset import load_dataset, prepare_y
    from asr import Speller, Attention

    (mapper, dataset, noisy_dataloader) = load_dataset('data/processed/index.tsv', 
        batch_size=2, n_jobs=1, use_gpu=False, text_only=True, drop_rate=0.2)

    loss_metric = torch.nn.CrossEntropyLoss(ignore_index=0, 
            reduction='none').to(torch.device('cpu'))

    speller = Speller(256, 256*2)
    attention = Attention(128, 256*2, 256)
    char_emb = nn.Embedding(dataset.get_char_dim(), 256)
    char_trans = nn.Linear(256, dataset.get_char_dim())

    text_autoenc = TextAutoEncoder(dataset.get_char_dim(), 
        speller=speller, attention=attention, char_emb=char_emb, char_trans=char_trans,
        tf_rate=0.9)

    optim = torch.optim.SGD(text_autoenc.parameters(), lr=0.01, momentum=0.9)

    for (y, y_noise) in noisy_dataloader:
        print(y.shape)

        y, y_lens = prepare_y(y)
        y_max_len = max(y_lens)
        
        print(y.shape)

        y_noise, y_noise_lens = prepare_y(y_noise)
        y_noise_max_lens = max(y_noise_lens)
        
        print("Clean string (length={}) :{}".format(y_lens[0] , dataset.decode(y[0, :].view(-1))))
        print("Noisy string (length={}) :{}".format(y_noise_lens[0], 
            dataset.decode(y_noise[0, :].view(-1))))
        
        print("Clean lengths: {}".format(y_lens))
        print("Noise lengths: {}".format(y_noise_lens))


        optim.zero_grad()

        # decode steps == longest target
        decode_step = y_max_len 
        noise_lens, enc_out = text_autoenc(y, y_noise, decode_step, 
            noise_lens=y_noise_lens)

        print(enc_out.shape)
        print(y.shape)
        
        b,t,c = enc_out.shape
        loss = loss_metric(enc_out.view(b*t,c), y.view(-1))
        # Sum each uttr and devide by length
        loss = torch.sum(loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
            .to(device = torch.device('cpu'), dtype=torch.float32)
        # Mean by batch
        loss = torch.mean(loss)


        print("LOSSS :::::: {}".format(loss))

        loss.backward()
        optim.step()

if __name__ == '__main__': 
    test_solver()