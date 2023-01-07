import numpy as np
import torch

class SeqData():
    """
    This class takes in a 1D equispaced time series data and generates 
    a bunch of samples of input and output sequences which 
    can be used for training an RNN model.
    """

    def __init__(self, data, num_samples, left_seq_size, right_seq_size,
                 skip_points, noise=0, quiet=False):

        """
        input data shape must be -> [n equispaced points of time series]

        Parameters
        ----------
        data: 1D float numpy array
            Origianl time series from which to get sequences
        num_samples: int
            Total number of sequence samples requested 
        left_seq_size: int
            Size of the input (or left) sequences; will be fed to the model
        right_seq_size: int
            Size of the output (or right) sequence; should be 1 for typical RNNs
            and greater than 1 for encoder-decoder type RNNs
        skip_points: int
            Number of points to skip; useful when the main data is very high resolution
        noise: float
            Scaling factor for a normal-type Noise to be added to the data
            noise * np.random.normal(0, 1, ....)
        quiet: Boolean
            Can be used to suppress print statements from this class

        Attributes
        ----------
        x: 2D float tensor array
            [left_seq_size,  num_samples] # inputs for RNN
        y: 2D float tensor array
            [right_seq_size, num_samples] # targets for RNN
        num_samples: int
            Total number of samples generated
        """

        main_data_shape = data.shape[0]

        # skip+1 since 0 is invalid and ::1 doesn't skip
        data = data[::skip_points+1]

        if not quiet and skip_points>0:
            print(f'Main data skip of {skip_points} has been applied!')
            print(f'Data shape changed from {main_data_shape} to {data.shape[0]} after skipping...')
            print('\n')

        if noise != 0:
            data += noise * np.random.normal(0, 1, data.shape[0])

        # convert to tensor data
        self.data = torch.from_numpy(data)
        
        self.x, self.y = self.__sequencyfy(self.data, num_samples,
                                           left_seq_size, right_seq_size)
        
        self.num_samples = self.x.shape[1]

        if not quiet:
            print(f'Dataset created with {self.x.shape[1]} samples. \n')
            print(f'Each sample sequence has {left_seq_size} input len')
            print(f'and {right_seq_size} target len.\n')

    def __sequencyfy(self, data, num_samples=100,
                     left_seq_size=50,right_seq_size=10):

        len_TS = data.shape[0]

        # to avoid index error when the index is too large to create
        # an input seq + output seq chunk
        rightmost_ind = len_TS - (left_seq_size + right_seq_size)

        # get a random numbers to pick iw+ow chunk
        rand_inds = torch.randint(0, rightmost_ind, (num_samples,))

        # initialize arrays to fill
        X = torch.zeros([left_seq_size,  num_samples], dtype=torch.float)
        Y = torch.zeros([right_seq_size, num_samples], dtype=torch.float)

        for i in range(num_samples):
            ls_start = rand_inds[i]
            X[:,i] = data[ls_start:ls_start + left_seq_size]

            rs_start = ls_start + left_seq_size
            Y[:,i] = data[rs_start:rs_start + right_seq_size]

        return X, Y
