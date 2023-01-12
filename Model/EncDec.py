# Author: Rakesh K. Yadav, 2023

import torch
import torch.nn as nn

class Encoder(nn.Module):

    """Encoder layer to encode a sequence to a hidden state"""

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
        super(Encoder, self).__init__()
        
        # store stuff in the class
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.flavor      = flavor

        if flavor=='rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        elif flavor=='lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif flavor=='gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, x): # x must be: [seq, batch_size]

        # gather weights in a contiguous memory location for 
        # more efficient processing
        self.rnn.flatten_parameters()
        
        # initialize hidden state with appropriate size
        h0 = torch.zeros(self.num_layers, x.size(1), 
                        self.hidden_size, device=x.device)

        if self.flavor=='lstm':
            # cell state only for lstm
            c0 = torch.zeros_like(h0)
            _, hidden = self.rnn(x.view(x.shape[0], x.shape[1], 
                                    self.input_size), (h0,c0))
        else:
            _, hidden = self.rnn(x.view(x.shape[0], x.shape[1], 
                                    self.input_size), h0)

        return hidden

class Decoder(nn.Module):

    """
        Decoder layer which uses a hidden state from the encoder layer 
        and makes predictions
    """

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
        super(Decoder, self).__init__()
        
        # store stuff in the class
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.flavor      = flavor

        if flavor=='rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        elif flavor=='lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif flavor=='gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        
        # will be used at the end of the RNN to do many-to-1 operation
        self.linear = nn.Linear(hidden_size, input_size)      

    def forward(self, x, encoder_hidden):

        # gather weights in a contiguous memory location for 
        # more efficient processing
        self.rnn.flatten_parameters()

        # x: [batch_size] is the end point of the primary input seq
        # view input as [1, batch_size, input_size] using unsqueeze
        out, hidden = self.rnn(x.unsqueeze(0), encoder_hidden)

        # out shape: [input_size(1), batch, hidden_size]
        out = self.linear(out.squeeze(0))     
        
        return out, hidden

class EncoderDecoder(nn.Module):

    """Combines the encoder and decoder classes to define a global model"""

    def __init__(self, encoder, decoder, npred):

        """
        Parameters
        ----------
        encoder: class
            RNN class that decodes a sequence to a hidden state
        decoder: class
            RNN class that takes in an encoder hidden state and 
            the last point of the input sequence to make predictions
        npred: int
            Number of points to predict
        """

        # inherit the nn.Module class via 'super'
        super(EncDec, self).__init__()
        
        # store stuff in the class
        self.enc   = encoder
        self.dec   = decoder
        self.npred = npred
        
    def forward(self, x): # x shape: [seq, batch]
        
        local_batch_size = x.shape[1]
        target_len = self.npred

        # convert to [seq, batch, 1]
        # 1 is for univariate sequence
        input_batch = x.unsqueeze(2)

        # initialize output array to be filled with predictions
        outs = torch.zeros(target_len, local_batch_size, 
                           input_batch.shape[2], device=x.device)

        # STEP 1: obtain the encoder hidden state for the inputs
        enc_hid = self.enc(input_batch)
        
        # STEP 2.1: grab last point of input batch for decoder
        dec_in = input_batch[-1, :, :] # shape: (batch_size, input_size)
        
        # STEP 2.2: assign the encoder hidden state to the decoder hidden
        dec_hid = enc_hid

        # STEP 3: make prediction like a traditional RNN point-by-point
        #         by using the predicted point as new input
        for t in range(target_len):
            # note that the dec_hid is being continuously rewritten
            dec_out, dec_hid = self.dec(dec_in, dec_hid)
            # store the prediction
            outs[t] = dec_out
            # feed back the prediction as input to the decoder
            dec_in =  dec_out
        
        return outs.reshape(target_len, local_batch_size)
