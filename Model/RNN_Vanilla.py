# Author: Rakesh K. Yadav, 2023


import torch.nn as nn
import torch

class RNNs(nn.Module):

    """Model class to declare an rnn and define a forward pass of the model."""

    def __init__(self, input_size, hidden_size, num_layers, flavor):

        """
        Parameters
        ----------
        input_size: int
            This is the same as number of features in traditional lingo.
            For univariate time series, this would be 1 and greater than 1
            for multivariate time series.
        hidden_size: int
            Number of hidden units in the RNN model
        num_layers: int
            Number of layers in the RNN model
        flavor: str
            Takes 'rnn', 'lstm', or 'gru' values.
        """

        # inherit the nn.Module class via 'super'
        super(RNN_Vanilla, self).__init__()

        # store stuff in the class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.flavor = flavor

        if flavor=='rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        elif flavor=='lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif flavor=='gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        
        # will be used at the end of the RNN to do many-to-1 operation
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x): 
        
        # rnn module expects data of shape [seq, batch_size, input_size]
        # since we have a univariate time series data, we need to add 
        # a dimension at the end
        x = x.unsqueeze(2) # shape now: [seq, batch_size, 1]

        batch_size = x.size(1)

        # initialize hidden state with appropriate size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        if self.flavor=='lstm':
            #cell state for lstm only
            c0 = torch.zeros_like(h0) 
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        # use the last index of RNN output
        out = out[-1,:,:] # shape: [batch_size, hidden_size]

        # run the data through a fully connected layer and return 1-point prediction
        # for each sequence in the batch
        return self.linear(out).reshape(1, batch_size)
