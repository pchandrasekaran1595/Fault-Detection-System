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

# ResNet50 Model; Slice out the final classification block and then Flatten
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.model = models.resnet50(pretrained=True, progress=True)
        self.model = nn.Sequential(*[*self.model.children()][:-1])
        self.model.add_module("Flatten", nn.Flatten())

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #

class Network(nn.Module):
    def __init__(self, IL=u.FEATURE_VECTOR_LENGTH, embed=None):
        nn.Module.__init__(self)

        # Input --> Embedding
        self.embedder = nn.Sequential()
        self.embedder.add_module("BN", nn.BatchNorm1d(num_features=IL, eps=1e-5))
        self.embedder.add_module("FC", nn.Linear(in_features=IL, out_features=embed))
        self.embedder.add_module("AN", nn.ReLU())

        # Embedding -> Similarity Predictor
        self.classifier = nn.Sequential()
        self.classifier.add_module("BN", nn.BatchNorm1d(num_features=embed, eps=1e-5))
        self.classifier.add_module("FC", nn.Linear(in_features=embed, out_features=1))

    # Setup Adam Optimizer
    def getOptimizer(self, lr=1e-3, wd=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    # Setup Dynamic Learn Rate Scheduler
    def getScheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x):
        return self.classifier(self.embedder(x))

# ******************************************************************************************************************** #

roi_extractor = RoIExtractor()
roi_extractor.to(u.DEVICE)
roi_extractor.eval()

fea_extractor = FeatureExtractor()
fea_extractor.to(u.DEVICE)
fea_extractor.eval()

# ******************************************************************************************************************** #

# Setup the Netowrk
def build_model(embed=None):
    if embed is not None:
        torch.manual_seed(u.SEED)
        model = Network(embed=embed)
    else:
        model = None
    
    lr, wd, batch_size = 1e-6, 1e-6, 512

    return model, batch_size, lr, wd

# ******************************************************************************************************************** #
