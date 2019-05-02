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
        self.decoder = SpeechDecoder(
            self.encoder.out_dim+listener_out_dim, 8*feature_dim)

    def forward(self, asr, x, x_len):
        '''
        Input arguments:
        * asr (nn.Module): An instance of the src.ASR class
        * x (tensor): A [batch_size, seq, feature] sized tensor, containing
        a batch of fbanks.
        * x_len (list): The lengths of each sample (unpadded), must be in
        descending order.
        '''

        listener_out, state_len = asr.encoder(x, x_len)

        # out shape: [batch, 1, 1, self.encoder.out_dim]
        encoder_out = self.encoder(x.unsqueeze(1))

        predict_seq = []
        # concatenate the two outputs
        for i in range(listener_out.shape[1]):
            decoder_in = torch.cat((listener_out[:, i, :].view(listener_out.shape[0], -1),
                encoder_out.view(encoder_out.shape[0], -1)), dim=1)

            # shape: [batch_size, 8*self.feature_dim]
            decoder_out = self.decoder(decoder_in)
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
        '''

        super(SpeechEncoder, self).__init__()

        assert len(ks) == 3 and len(num_filters) == 3
        
        self.out_dim = num_filters[-1]

        # in : (batch_size, seq, feature_dim)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters[0],
                kernel_size=ks[0]),
            nn.BatchNorm2d(num_features=num_filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(pool_ks[0]))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=ks[1]),
            nn.BatchNorm2d(num_features=num_filters[1]),
            nn.ReLU(),
            nn.MaxPool2d(pool_ks[1]))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=ks[2]),
            nn.BatchNorm2d(num_features=num_filters[2]),
            nn.ReLU(),
            nn.MaxPool2d(pool_ks[2]))

    def forward(self, x):
        '''
        Input arguments: 
        * x:  A [batch_size, seq, feature_dim] tensor filterbank
        '''
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
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

        self.leaky_1 = nn.LeakyReLU()
        self.leaky_2 = nn.LeakyReLU()
        self.out = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        '''
        Input arguments:
        * x: A [batch_size, self.in_dim] shaped tensor which is a concatted
        representation of the global speech encoder output and a single frame
        from the listener output.

        Output: A [batch_size, self.out_dim], where out dim is reshapable into 8
        original frames of input fbanks.
        '''
        x = self.leaky_1(x)
        x = self.leaky_2(x)
        x = self.out(x)
        return x

def test_solver():
    from asr import Listener
    from dataset import load_dataset, prepare_x, prepare_y

    (mapper, dataset, dataloader) = load_dataset('data/processed/index.tsv', 
        batch_size=2, n_jobs=1, use_gpu=False)


    kernel_sizes = [(36, 1), (1, 5), (1, 3)]
    num_filters = [32, 64, 256]
    # HACK: I set the final max pooling kernel sizes at just some high value to cover
    # the whole thing
    max_pooling_sizes = [(3, 1), (5, 1), (2000, 40)]

    # TODO: add to device, like line 80 in solver.py
    loss_metric = nn.SmoothL1Loss(reduction='none')

    listener = Listener(256, 40)
    speech_autoenc = SpeechAutoEncoder(kernel_sizes, num_filters, 
        max_pooling_sizes, 40, listener)

    optim = torch.optim.SGD(speech_autoenc.parameters(), lr=0.01, momentum=0.9)

    for x, y in dataloader:
        
        x, x_lens = prepare_x(x)

        optim.zero_grad()

        enc_out = speech_autoenc(x, x_lens)

        # pad the encoder output UP to the maximum batch time frames and 
        # pad x DOWN to the same number of frames
        batch_t = max(x_lens)
        x = x[:, :batch_t, :]
        enc_final = torch.zeros([enc_out.shape[0], batch_t, enc_out.shape[2]])
        enc_final[:, :enc_out.shape[1], :] = enc_out
    
        loss = loss_metric(enc_final, x)

        # Divide each by length of x and sum, then take the average over the
        # batch
        loss = torch.sum(loss.view(loss.shape[0], -1)) / torch.Tensor(x_lens)
        loss = torch.mean(loss)

        # backprop
        loss.backward()
        optim.step()


if __name__ == '__main__':
    test_solver()