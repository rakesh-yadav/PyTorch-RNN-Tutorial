# PyTorch tutorial for using RNNs and Encoder-Decoder RNNs for time series forcasting and hyperparameter tuning

## Some blabber

Hi there! I am glad you decided to stop by this corner of the internet :grinning:

This package resulted from my effort to write a simple PyTorch based ML package that uses recurrent neural networks (RNN) to predict a given time series data. 

You must be wondering why you should bother with this package since there is a lot of stuff on the internet on this topic. Well, let me tell you that I have traveled the internet lanes and I was really frustrated by how scattered the information is in this context. It was a lot of effort to collect all the relevant parts from the internet and construct this package. 

I had only a basic background in ML and zero knowledge of PyTorch (using Keras doesn't prepare you for PyTorch :stuck_out_tongue:) when I started writing this package. But that actually ended up being a blessing in disguise. Since I was starting from scratch, I was able to write the code in a way that was intuitive and easy to understand for people who are new to the subject.

So if you're feeling lost and frustrated, give this package a try. It might just help you understand not only RNNs, but PyTorch as well. And who knows, you might even have a little fun along the way.

## Code Functionalities
1. Many-to-One prediction using PyTorch's vanilla versions of RNN, LSTM, and GRU.
2. Many-to-Many (or Seq2Seq) prediction using Encoder-Decoder architecture; base units could be RNN, LSTM, or GRU.
3. Hyperparameter Tuning! It uses the [Optuna](https://optuna.org/) library for that.
4. Save PyTorch models, as well as reload and train them further.
5. Works on any **univariate** and **equispaced** time series data.
6. Can use GPUs.

## Usage
Best way to figure out how to use this package is to check out the example notebooks available in the `Notebooks` folder.

I have also made a sample notebook available in Google Colab!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dsP3FhY-qghqfcmn6TLaUkuyCaWAl8WL?usp=sharing)

## Code Structure
* `Model`: model definition classes. 

* `Notebooks`: example notebooks for code usage and for performing hyperparameter tuning.

* `Saved_models`: empty; directory where output from the code is saved.

* `Utils` folder has three class files:
    * Create_and_Train.py is THE main file which creates a model, runs the epoch loop, saves models and loss curves.
    * Trainer.py contains the training loop definition, a _test_ function to run the model on test data, as well as functions to make predictions. 
    * SeqData.py file is used to create sequenced dataset given a 1D time series.

* `imports.py` file is used by the notebooks present in the `Notebooks` folder.

* `requirements.txt` file can be used in conjunction with pip to install the required packages.

### Note 
I also recommend checking out my colleague's [implementation](https://github.com/lkulowski/LSTM_encoder_decoder) of rnn in pytorch.
