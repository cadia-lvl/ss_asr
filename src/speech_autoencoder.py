import torch
import torch.nn as nn
import torch.functional as F

class SpeechAutoEncoder(nn.Module):
    def __init__(self, listener_out_dim, feature_dim, kernel_sizes, num_filters, 
        pool_kernel_sizes):
        '''
        Input arguments:
        * kernel_sizes (List): A list of kernel sizes (only 3 values) for
        the 3 convolutional layers
        * num_filters (List): A list of output filters (only 3 values) 
        for the 3 convolutional layers
        * pool_kernel_sizes (List): A list of kernel sizes (only 3 values) for
        the max pooling of each convolutional layer
        * feature_dim (int): The dimensionality of the fbanks
        * listener_out_dim (int): The output dimensionality of the 
        listener module
        '''
        super(SpeechAutoEncoder, self).__init__()

        self.feature_dim = feature_dim
        self.encoder = SpeechEncoder(kernel_sizes, num_filters, pool_kernel_sizes)
        '''
        The decoder takes as input concatenation of all global 
        encoder output and one step of listener output. It predicts
        8 frames of input audio, hence 8 * feature_dim
        '''        
        self.decoder = SpeechDecoder(
            self.encoder.out_dim+listener_out_dim, 8*feature_dim)

    def forward(self, x, listener_out, just_first=False):
        '''
        Input arguments:
        * asr (nn.Module): An instance of the src.ASR class
        * x (tensor): A [batch_size, seq, feature] sized tensor, containing
        a batch of padded fbanks.
        * listener_out (tensor): A [batch_size, ~seq/8, feature] sized tensor
        that is the output of the ASR encoder
        * just_first (bool): Can be used for debugging. If set to true, only
        the first 8 frames of the ground truth are returned.
        
        Steps:
        1. Given X, the global encoder will encode the whole utterance of each sample,
        shaped e.g [bs, global_enc_out]

        2. The input of the decoder is the concat of the whole global encoder output and
        a single step from the listener encoder, shaped [bs, global_enc_out + lis_out]
        For the whole batch, we will therefore have created ~seq/8 new batches that
        are passed on to the decoder

        '''
        # global-encode the whole batch of utterances
        # encoder_out shape: [batch, 1, 1, self.encoder.out_dim]
        encoder_out = self.encoder(x.unsqueeze(1)) # x is now [bs, 1, seq, features]
        # reshape into [batch, encoder.out_dim]
        
        predict_seq = []
        # concatenate the two outputs
        num = listener_out.shape[1]
        if just_first:
            num = 1
        for i in range(num):
            '''
            Decoder in is the concatenation of a single listener output
            and the complete global-encoder output:

            [l_1, l_2, ..., l_k, g_1, g_2, ... g_m]
            '''
            
            # shape : [batch_size, listener.out_dim]
            listener_step = listener_out[:, i, :]
            # shape : [batch_size, encoder.out_dim + listener.out_dim]
            decoder_in = torch.cat((listener_step, encoder_out), dim=1)

            # shape: [batch_size, 8*self.feature_dim]
            decoder_out = self.decoder(decoder_in)
            
            '''
            This suspicious .view() has been verified, that is
            decoder_out[i, j, k] represents the k-th frequency band
            of the j-th frame in the i-th sample
            '''
            # shape: [batch_size, 8, self.feature_dim]
            decoder_out = decoder_out.view(decoder_out.shape[0], 8, self.feature_dim)
            
            predict_seq.append(decoder_out)

        # shape: [batch_size, 8*(~batch_seq/8), self.feature_dim] where batch_seq
        # is the maximum length of the current batch (comes about b.c. of packing) 
        out = torch.cat(predict_seq, dim=1)
        return out


class SpeechEncoder(nn.Module):
    def __init__(self, ks, num_filters, pool_ks):
        '''
        Input arguments:
        * ks (List): A list of kernel sizes (only 3 values) for
        the 3 convolutional layers
        * num_filters (List): A list of output filters (only 3 values) for
        the 3 convolutional layers
        * pool_ks (List): A list of kernel sizes (only 3 values) for
        the max pooling of each convolutional layer

        kernel_sizes: [[36, 1], [1, 5], [1, 3]]    
        num_filters: [32, 64, 256]      
        pool_kernel_sizes: [[3, 1], [5, 1], [2000, 40]]

        '''
        super(SpeechEncoder, self).__init__()
        assert len(ks) == 3 and len(num_filters) == 3
        self.out_dim = num_filters[-1]

        # in : (batch_size, seq, feature_dim)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters[0],
                kernel_size=ks[0],
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_features=num_filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(pool_ks[0]))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=ks[1],
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_features=num_filters[1]),
            nn.ReLU(),
            nn.MaxPool2d(pool_ks[1]))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=ks[2],
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_features=num_filters[2]),
            nn.ReLU(),
            nn.MaxPool2d(pool_ks[2]))

    def forward(self, x):
        '''
        Input arguments: 
        * x:  A [batch_size, seq, feature_dim] tensor filterbank
        Returns:
        * x: A [batch_size, self.out_dim, 1, 1]
        '''
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        # x shape : [bs, out_dim, 1, 1]
        # squeeze out the last two
        x = x.squeeze(2).squeeze(2)
        return x
        
class SpeechDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        Input arguments:
        * in_dim (int): The dimensionality of each input sample
        * out_dim (int): The dimensionality of each ouput sample.
        This dimension should be reshapable into 8 orginal frames
        of fbanks.

        The speech decoder is a simple feed-forward neural 
        network with two leaky ReLU layers and a linear layer 
        on top. 
        It takes as input a single vector, a concatenation of 
        the output from the global encoder and a single output 
        from the ASR encoder

        It generates eight frames of output speech features which
        are scored against the eight frames of input speech features 
        that produced the given ASR encoder output.
        
        (Expected output: [batch_size, 8, feature_dim])
        '''
        super(SpeechDecoder, self).__init__()
        self.core = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, out_dim))
    
    def forward(self, x):
        '''
        Input arguments:
        * x: A [batch_size, self.in_dim] shaped tensor which is a concatted
        representation of the global speech encoder output and a single frame
        from the listener output.

        Output: A [batch_size, self.out_dim], where out dim is reshapable into 8
        original frames of input fbanks.
        '''
        return self.core(x)