"""
    Script to hold the Dataset Templates used by the Application
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import utils as u

# ******************************************************************************************************************** #

# Used in Feature Extraction
class FEDS(Dataset):
    def __init__(self, X=None, transform=None):
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.X[idx])

# ******************************************************************************************************************** #

"""
    All other Dataset Templates
"""