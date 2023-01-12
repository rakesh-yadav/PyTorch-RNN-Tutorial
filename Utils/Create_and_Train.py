# Author: Rakesh K. Yadav, 2023


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from .Trainer import *

class Create_and_Train():

    """
        This class performs model declaration, training, testing, validation
        and saving of model and loss curves. 
    """

    def __init__(self, params, data, device):

        """
        Parameters
        ----------
        params: dict
            Contains all the model specifications and a bunch of other stuff
                'train':          bool,   To train or not to train
                'EPOCHS':         int,    Number of epochs
                'num_layers':     int,    Number of layer in the model
                'hidden_size':    int,    Number of hidden units in the model 
                'batch_size':     int,    Batch size during the epoch training
                'input_size':     int,    Number of features; 1 for univariate time series
                'learning_rate':  float,  Learning rate for training; typical value 0.001
                'flavor':         str,    Which RNN model to use. 'rnn', 'lstm', 'gru'
                'left_seq_size':  int,    Size of the input sequence
                'right_seq_size': int,    Size of prediction. 1 for RNN and >1 for Encoder-Decoder
                'noise':          float,  Noise factor to use; Put it to 0.0 if no noise needed
                'norm_fac':       float,  The time series is normalized by data max before it is given
                                          to the model. I am storing it to keep track of it.
                'save_path':      str,    Path of dir where model and loss curves will be saved.
                'load_previous':  bool,   If True, it loads a previously run model and trains it further
                'encdec':         bool,   If True, it constructs an Encoder-Decoder model using 
                                          the 'flavor' chosen above
                'test':           bool,   If True, it performs testing on test data and plots on 
                                          the loss curve
                'val':            bool,   If True, it predicts for validation data and saves a .npz file
                                          which contains the error between targets and predictions
                'save_model':     bool,   If True, saves modes whenever the test loss decreases
                'iplot':          bool    If True, saves the loss curves as PNG files
        
        data: dict
            Contains Xtrain, Ytrain, Xtest, Ytest, Xval, Yval

        device: str
            'cuda' or 'cpu'
        """

        self.params = params
        self.device = device
        
        # define model and trainer
        self.model = self.__define_model(params)
        self.model.to(device)

        # define the optimizer
        self.optimizer = optim.AdamW(self.model.parameters(),
                                    amsgrad=True, 
                                    lr=params['learning_rate'], 
                                    weight_decay=0) #alpha parameter for l2 regularization

        # feed the model and optimizer to the Trainer class; see Trainer.py
        self.trainer = Trainer(self.model, self.optimizer)

        self.first_epoch = 1  # will be updated if load_previous model is true

        if params['save_model']:
            # construct a descriptive file name
            self.model_filename = (f'L{params["num_layers"]}_'\
                                   f'H{params["hidden_size"]}_'\
                                   f'lr{params["learning_rate"]}_'\
                                   f'input{params["left_seq_size"]}_'\
                                   f'target{params["right_seq_size"]}_'\
                                   f'noise{params["noise"]}')

        if params['load_previous']:
            self.__load_previous(params)

        # loss below which model should be saved
        self.best_loss = 0.1

        # load appropriate tqdm depending on the environment
        if self.__is_notebook(): import tqdm.notebook as tqbar
        else: import tqdm as tqbar
        
        # declare the tqdm progress bar
        epoch_bar = tqbar.tqdm(range(params['EPOCHS']),
                               desc="Epochs progress [Loss: NA]",
                               unit='Epoch')

        #--------------------Main Train loop---------------------------------
        if params['train']:

            # ---------Send data to device----------
            Xtrain = data['Xtrain'].to(device)  # shape: [seq, samples]
            Ytrain = data['Ytrain'].to(device)  # shape: [seq, samples]

            if params['test']:
                Xtest = data['Xtest'].to(device)  
                Ytest = data['Ytest'].to(device) 
            if params['val']:
                Xval = data['Xval'].to(device)
                Yval = data['Yval'].to(device)

            # allocate loss arrays to be filled later    
            self.train_losses = np.full(params['EPOCHS'], np.nan)
            if params['test']: 
                self.test_losses = np.full(params['EPOCHS'], np.nan)
            if params['val']:
                self.val_err = torch.zeros( params['EPOCHS'], 
                                            Yval.shape[0],
                                            Yval.shape[1],
                                            device=Yval.device)
                
            # -------Epoch loop------
            for epoch in epoch_bar:
                # get train loss for an epoch
                self.train_loss = self.trainer.train(Xtrain,Ytrain,params['batch_size'])

                # store the training loss
                self.train_losses[epoch] = self.train_loss

                # Get test and val loss and errors, if asked
                if params['test']:
                    self.test_loss  = self.trainer.test(Xtest,Ytest)
                    self.test_losses[epoch] = self.test_loss
                if params['val']:
                    if params['encdec']:
                        # get mean square error for validation data
                        self.val_err[epoch] = (Yval - self.trainer.pred_encdec(Xval).detach())**2
                    else:
                        # get mean square error for validation data
                        self.val_err[epoch] = (Yval - self.trainer.RNN_npoint_pred(Xval, Yval.shape[0]))**2

                # update loss curve on every 10th epoch
                if params['iplot'] and ((epoch+1) % 10)==0:
                    self.__plot_loss_curve(params)

                # update saved good model
                if params['save_model'] and params['test']:
                    if self.test_loss < self.best_loss: 
                        self.__save_model(epoch, params)
                        #update loss to compare later
                        self.best_loss = self.test_loss
                elif params['save_model'] and self.train_loss < self.best_loss:
                    self.__save_model(epoch, params)
                    self.best_loss = self.train_loss

                # update tqdm epoch progress bar
                epoch_bar.set_description('Epochs progress [Loss: {:.3e}]'.format(self.train_loss))
        else:
            print(f'Nothing to train since Training is set to {params["train"]}....')
        #---------------------------------------------------------------

        #plot final loss curve
        if params['train'] and params['iplot']:
            self.__plot_loss_curve(params)

        #-----Save nPoint prediction error mean and std
        if params['train'] and params['val']:
            os.chdir(params['save_path'])
            # val_err shape: [epochs, seqs, batch]
            val_err_mean = torch.mean(self.val_err, dim=2) # along batch
            val_err_std = torch.std(self.val_err, dim=2)   # along batch
            self.err_filename = (self.model_filename+
                                f'_nPoint_err_epochs{self.first_epoch}to{self.first_epoch+params["EPOCHS"]-1}')
            np.savez_compressed(self.err_filename,
                                mean=val_err_mean.detach().cpu(),
                                std=val_err_std.detach().cpu())
            
            
            
    #---------------Helper functions-----------------  
    def __define_model(self, params):
        sys.path.append('../Model')
        
        if params['encdec']:
            from Model.EncDec import Encoder, Decoder, EncoderDecoder
            enc = Encoder(params['input_size'], params['hidden_size'], params['num_layers'], params['flavor'])
            dec = Decoder(params['input_size'], params['hidden_size'], params['num_layers'], params['flavor'])
            model = EncDec(enc, dec, params["right_seq_size"])
            print('**************************************************************************')
            print(f'EncDec {params["flavor"]} Regression model initialized with '\
                  f'{params["num_layers"]} layers and {params["hidden_size"]} hidden size.')
            print(f'I will take in {params["left_seq_size"]} points and predict {params["right_seq_size"]} points.')
            print('**************************************************************************')
        else:
            from Model.RNN_Vanilla import RNNs
            model = RNNs(params['input_size'], params['hidden_size'], params['num_layers'], params['flavor'])
            print('**************************************************************************')
            print(f'RNN {params["flavor"]} Regression model initialized with '\
                  f'{params["num_layers"]} layers and {params["hidden_size"]} hidden size.')
            print(f'I will take in {params["left_seq_size"]} points and predict {params["right_seq_size"]} points')
            print('**************************************************************************')
            
            if params["right_seq_size"]>1:
                sys.exit('!!! ERROR: Traditional RNNs can not predict more than 1 point. Adjust target size...')

        return model

    def __load_previous(self, params):
        os.chdir(params['save_path'])
        files = glob.glob(self.model_filename+'_epoch*.pth')
        # look for the most recent file
        files.sort(key=os.path.getmtime)
        if len(files)>0: 
            print('Found older file:', files[-1])
            print('Loading.....')
            checkpoint = torch.load(files[-1], map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # get the epoch of the saved model
            last_epoch = checkpoint['epoch']
            # now update the 'first' epoch which can be used in 
            # a new filename
            self.first_epoch = last_epoch+1

    def __save_model(self, epoch, params):
        os.chdir(params['save_path'])
        print_loss = self.test_loss if self.params['test'] else self.train_loss
        # use '\r' ending to overwrite the printed message
        print('Lowest loss ({:0.3e}) decreased. Saving model....'.format(print_loss), end='\r')
        filename =self.model_filename+f'_epochs{self.first_epoch}to{self.first_epoch+self.params["EPOCHS"]-1}'+'.pth'
        torch.save({'params': self.params,          # save model definition dict
                    'epoch': self.first_epoch+epoch,# save the epoch of the model
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, # crucial for restarting training
                     filename)

    def __plot_loss_curve(self, params):
        os.chdir(params['save_path'])
        fig = plt.figure(figsize=(8, 4), num=1, clear=True)
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        xdata = self.first_epoch+np.arange(self.params['EPOCHS'])
        ax.semilogy(xdata, self.train_losses, 'r', label='Train')
        if self.params['test']: ax.plot(xdata, self.test_losses, 'g', label='Test')   
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        plt.legend()
        fig_name = self.model_filename+f'_LossCurve_epochs{self.first_epoch}to{self.first_epoch+self.params["EPOCHS"]-1}.png'
        plt.savefig(fig_name, dpi=150)

    def __is_notebook(self):
        #credit -> https://stackoverflow.com/a/39662359
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter
