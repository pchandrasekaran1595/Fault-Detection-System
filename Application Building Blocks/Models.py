"""
    Models Used
"""

import torch
from torchvision import models
from torch import nn, optim
import utils as u

# ******************************************************************************************************************** #

# Region-of-Interest Extractor (Object Detector)
class RoIExtractor(nn.Module):
    def __init__(self):
        super(RoIExtractor, self).__init__()
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True)

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #

# VGG16 Model; Slice out the final 2 blocks and Average Pool the 512x7x7 features down to 512x2x2 and then Flatten
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.model = models.vgg16_bn(pretrained=True, progress=True)
        self.model = nn.Sequential(*[*self.model.children()][:-2])
        self.model.add_module("Adaptive Avg Pool", nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        self.model.add_module("Flatten", nn.Flatten())

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #


class Your_Network_Here(nn.Module):
    def __init__(self, IL=u.FEATURE_VECTOR_LENGTH, embed=None):
        super(Your_Network_Here, self).__init__()

        """
            .
            .
            .
            .
        """

    # Setup Adam Optimizer
    def getOptimizer(self, lr=1e-3, wd=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    # Setup Dynamic Learn Rate Scheduler
    def getScheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #

roi_extractor = RoIExtractor()
roi_extractor.to(u.DEVICE)
roi_extractor.eval()

fea_extractor = FeatureExtractor()
fea_extractor.to(u.DEVICE)
fea_extractor.eval()

# ******************************************************************************************************************** #

# Setup the Siamese Netowrk
def build_your_model(*args, **kwargs):
    if some *args or **kwargs is not None:
        torch.manual_seed(u.SEED)
        model = Your_Network_Here(*args, *kwargs)
    else:
        model = None

    lr = 1e-6
    wd = 1e-6
    batch_size = 512

    return model, batch_size, lr, wd

# ******************************************************************************************************************** #
