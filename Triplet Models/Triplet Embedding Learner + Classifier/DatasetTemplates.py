import numpy as np
import torch
from torch.utils.data import Dataset

# ******************************************************************************************************************** #

# Dataset Template used for Feature Extraction
class FEDS(Dataset):
    def __init__(self, X=None, transform=None):
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.X[idx])

# ******************************************************************************************************************** #

# Dataset Template used to generate data that can be passed to the Triplet Embedder Learner
class TripletDS(Dataset):
    def __init__(self, anchor, p_vector, n_vector):
            self.anchor = anchor
            self.p_vector = p_vector
            self.n_vector = n_vector
            
    def __len__(self):
        # return self.p_vector.shape[0]
        return self.n_vector.shape[0]
    
    def __getitem__(self, idx):
        return self.anchor, self.p_vector[idx], self.n_vector[idx]

# ******************************************************************************************************************** #

# Dataset Template used to generate data that can be passed to the Classifier
class DS(Dataset):
        def __init__(self, p_vector=None, n_vector=None):
            self.X = np.concatenate((p_vector, n_vector), axis=0)
            self.y = np.concatenate((np.ones((p_vector.shape[0], 1)), np.zeros((n_vector.shape[0], 1))), axis=0)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

# ******************************************************************************************************************** #
