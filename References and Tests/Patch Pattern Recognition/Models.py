"""
    Script that holds the Deep Learning Model. (Uses Pytorch)
"""

import torch
from torch import nn
from torchvision import models, transforms

# ******************************************************************************************************************** #

# Normalize the vector to a min-max of [0, 1]
def normalize(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x

# ******************************************************************************************************************** #

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Device on which to run inference on. (This is now platform aware; will choose NVIDIA GPU if present, else will run on CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformations on Image Data Expected by the Model
        # 1. ToTensor(): Converts data to a torch tesnor and converts if from np.uint8 format to floating point values between (0, 1)
        # 2. Normalize() : ImageNet Normalization
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),])

        # Get the model
        self.model = models.vgg16_bn(pretrained=True, progress=True)

        # Cut out the last 2 Layers
        self.model = nn.Sequential(*[*self.model.children()][:-2])

        # Adaptive Average Pool to an output size of 512x2x2 from 512x7x7
        self.model.add_module("Adaptive Average Pool", nn.AdaptiveAvgPool2d(output_size=(2, 2)))

        # Flatten the output
        self.model.add_module("Flatten", nn.Flatten())
    

    def forward(self, x):
        return self.model(x)
    

    # Extract Features from a single image
    def get_features(self, image):
        # load model onto the device
        self.to(self.device)

        # Extract features
        # Always use torch.no_grad() or torch.set_grad_enabled(False) when performing inference (or during validation)
        with torch.no_grad():
            features = self(self.transform(image).to(self.device).unsqueeze(dim=0))

        # Return Normalized Features
        return normalize(features)
    

    # Extract features from batches of images. Expects a pytorch dataloader to be passed as an argument
    def get_batch_features(self, dataloader):
         # load model onto the device
        self.to(self.device)

        # Create a empty tensor to which batches of features can be appended
        features = torch.empty((0, 2048), device=self.device)

        # Iterate through the dataloader
        for X in dataloader:

            # Load tensor onto the mdoel
            X = X.to(self.device)

            # Perform inference on batch
            with torch.no_grad():
                output = self(X)
            
            # Concatenate batch
            features = torch.cat((features, output), dim=0)
        
        # Return Normalized Features
        return normalize(features)


    # Function to return the cosine similarity between 2 Pytorch Tensors
    def get_cosine_similarity(self, features_1, features_2):
        return nn.CosineSimilarity()(features_1, features_2).item()

# ******************************************************************************************************************** #

def build_model():
    model = FeatureExtractor()
    model.eval()

    return model

# ******************************************************************************************************************** #
