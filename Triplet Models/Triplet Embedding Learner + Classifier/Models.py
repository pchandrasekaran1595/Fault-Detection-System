import os, torch
from torchvision import models
from torch import nn, optim
import utils as u

# ******************************************************************************************************************** #

class RoIExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True)

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #

class FeatureExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.model = models.vgg16_bn(pretrained=True, progress=True)
        self.model = nn.Sequential(*[*self.model.children()][:2])
        self.model.add_module("Adaptive Avg Pool", nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        self.model.add_module("Flatten", nn.Flatten())

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #

class EmbeddingNetwork(nn.Module):
    def __init__(self, IL=u.FEATURE_VECTOR_LENGTH, embed=None):
        super(EmbeddingNetwork, self).__init__()

        self.embedder = nn.Sequential()
        self.embedder.add_module("BN", nn.BatchNorm1d(num_features=IL, eps=1e-5))
        self.embedder.add_module("FC", nn.Linear(in_features=IL, out_features=embed))
        self.embedder.add_module("AN", nn.ReLU())
    
    def getOptimizer(self, lr=1e-3, wd=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x1, x2=None, x3=None):
        if x2 is not None and x3 is not None:
            x1 = self.embedder(x1)
            x2 = self.embedder(x2)
            x3 = self.embedder(x3)
            return x1, x2, x3
        else:
            return self.embedder(x1)

# ******************************************************************************************************************** #

class Network(nn.Module):
    def __init__(self, embedding_net=None, embed=None):
        super(Network, self).__init__()

        self.embedding_net = embedding_net
    
        self.classifier = nn.Sequential()
        self.classifier.add_module("BN", nn.BatchNorm1d(num_features=embed, eps=1e-5))
        self.classifier.add_module("FC", nn.Linear(in_features=embed, out_features=1))
    
    def getOptimizer(self, lr=1e-3, wd=0):
        p = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(p, lr=lr, weight_decay=wd)

    def forward(self, x):
        return self.classifier(self.embedding_net(x))

# ******************************************************************************************************************** #

roi_extractor = RoIExtractor()
roi_extractor.to(u.DEVICE)
roi_extractor.eval()

fea_extractor = FeatureExtractor()
fea_extractor.to(u.DEVICE)
fea_extractor.eval()

# ******************************************************************************************************************** #

def build_embedder(embed=None):
    if embed is not None:
        torch.manual_seed(u.SEED)
        model = EmbeddingNetwork(embed=embed)
    else:
        model = None

    lr = 1e-6
    wd = 1e-6
    batch_size = 512

    return model, batch_size, lr, wd

# ******************************************************************************************************************** #

def build_classifier(embedding_net=None, path=None, embed=None):
    if embed is not None:
        embedding_net.load_state_dict(torch.load(os.path.join(path, "embedder_state.pt"), map_location=u.DEVICE)["model_state_dict"])
        embedding_net.eval()
        for params in embedding_net.parameters():
            params.requires_grad = False
    
        torch.manual_seed(u.SEED)
        model = Network(embedding_net=embedding_net, embed=embed)
    
    else:
        model = None

    lr = 1e-3
    wd = 1e-5
    batch_size = 512

    return model, batch_size, lr, wd

# ******************************************************************************************************************** #
