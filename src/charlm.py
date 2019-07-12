import torch
import torch.nn as nn
from torch.autograd import Variable

class CharLM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CharLM, self).__init__()
        '''
        Input arguments:
        * input_size (int): The dimension of input samples (character dim)
        * hidden_size (int): A freely selected hyperparameter, determines
        the size of hidden features in RNN cells.

        Each batch is of shape [batch_size, chunk_size, char_dim]
        and CharRNN receives at each timestep as input a tensor of shape
        [batch_size, 1, char_dim] representing a single input character, x_t,
        from each batch. CharRNN tries to predict x_{t+1}.
        
        The output dimensionality of the network is equal to the input 
        dimensionality
        
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(input_size, hidden_size)

        '''
        Input to GRUCell is:
        * input (batch_size, input_size) containing input features
        * hidden (batch_size, hidden_size) containing previous hidden
        features
        Output of GRUCell is:
        * hidden (batch_size, hidden_size) containing the current hidden
        state emission
        '''
        self.layer_1 = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size)
        self.layer_2 = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size)

        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x, h_1, h_2):
        '''
        Input arguments:
        x: A [batch_size] tensor, containing the character input for each
        string in the batch
        hidden: The previous hidden state
        '''
        x = self.emb(x.long())
        h_1 = self.layer_1(x, h_1) # expects input of shape [bs, input_size]
        h_2 = self.layer_2(h_1, h_2)
        out = self.out(h_2)
        return out, (h_1, h_2)
        
    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_size).to(device),
            torch.zeros(batch_size, self.hidden_size).to(device))