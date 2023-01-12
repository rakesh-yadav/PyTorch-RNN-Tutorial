# Author: Rakesh K. Yadav, 2023


import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from Utils import *
from Model import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using: {device}. Device: {torch.cuda.get_device_name()}')
