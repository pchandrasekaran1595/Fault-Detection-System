import numpy as np
import torch
from torch.utils.data import Dataset

# ******************************************************************************************************************** #

class FEDS(Dataset):
    def __init__(self, X=None, transform=None):
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.X[idx])

# ******************************************************************************************************************** #

class DS(Dataset):
    def __init__(self, p_vector=None, n_vector=None):
        self.X = np.concatenate((p_vector, n_vector), axis=0)
        self.y = np.concatenate((np.ones((p_vector.shape[0], 1)), np.zeros((n_vector.shape[0], 1))), axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

# ******************************************************************************************************************** #
