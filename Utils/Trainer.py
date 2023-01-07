# Author: Rakesh K. Yadav, 2023


import torch
import torch.nn as nn

class Trainer:

    """Contains the training loop definition, and a few other functions to perform 
        testing on data and predict using a trained model.    
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
    def loss(self, outputs, targets):
        # using the 'sum' argument to obtain the batch summed
        # and points summed, if using >1 predictions in EncoderDecoder,
        # loss. When reporting I'll devide by batch size and the 
        # number of points manually
        return nn.MSELoss(reduction='sum')(outputs, targets)
    
    def train(self, Xtrain, Ytrain, batch_size):
        # Xtrain: [seq,      total_train_data] 
        # Ytrain: [1 or seq, total_train_data]
        self.model.train()

        # declare random arrray each time the function is called 
        # in the epoch loop. Will be used to randomize the 
        # training data
        shuff_inds = torch.randint(0, Xtrain.shape[1], (Xtrain.shape[1],))

        total_loss = 0

        # Note: if batch_size in not an integer multiple of
        # traini_size, the loop below only uses the last remaining
        # chunk of the data which will be smaller than the batch_size
        # Apparently a property of python 
        for batch in range(0,Xtrain.shape[1],batch_size):
            self.optimizer.zero_grad()
            indices = shuff_inds[batch:batch+batch_size]
            outputs = self.model(Xtrain[:,indices])
            loss = self.loss(outputs, Ytrain[:,indices])
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()

        #return per sample and per prediction point loss
        return total_loss/Ytrain.shape[0]/Ytrain.shape[1]
        
    
    def test(self, Xtest, Ytest):
        self.model.eval()
        loss = self.loss(self.model(Xtest), Ytest)
        return loss.item()/Ytest.shape[0]/Ytest.shape[1]

    def pred_encdec(self, x):
        # only valid for encoder-decoder model
        self.model.eval()
        outs = self.model(x)
        return outs
    
    def RNN_npoint_pred(self, input_seqs, n_pred):
        # input_seqs : [seqs, batch]
        # n_pred: number of points to predict using simple RNN
        self.model.eval()

        input_seq_size = input_seqs.shape[0]
        input_batch = input_seqs

        # array used for updating the input_batch
        # in the loop below
        preds = input_seqs

        for i in range(n_pred):
            # get 1 point prediction
            pred = self.model(input_batch)

            # stop gradient track by "detach" and then 
            # attach the point to the end of input seq
            preds = torch.concat((preds, pred.detach()), dim=0)

            # update the input to RNN to include the new prediction point
            input_batch = preds[-input_seq_size:, :]

        # now return the last n_pred points
        return preds[-n_pred:,:] #shape: [n_pred, batch]
    
