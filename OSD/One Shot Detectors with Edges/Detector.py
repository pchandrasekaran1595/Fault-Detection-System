import cv2
import platform
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

import utils as u

# ******************************************************************************************************************** #

# Function to normalize values in a vector of length 'n' to min:0 and Max:1
def normalize(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x

# ******************************************************************************************************************** #

# Function to build the model
def build_model():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            # Size of Image expected by the model
            self.size = 224

            # Device on which to run inference on. (This is now platform aware; will choose NVIDIA GPU if present, else will run on CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Transformations on Image Data Expected by the Model
            # 1. ToTensor(): Converts data to a torch tensor and converts if from np.uint8 format to floating point values between (0, 1)
            # 2. Normalize() : ImageNet Normalization
            self.transform = transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),])

            # Get the model
            self.model = models.vgg16_bn(pretrained=True, progress=True)

            # Cut out the Adaptive Average Pool and Classifier
            self.model = nn.Sequential(*[*self.model.children()][:-2])

             # Add in a new Adaptive Average Pool of size (2,2)
            self.model.add_module("Adaptive Avgpool", nn.AdaptiveAvgPool2d(output_size=(2, 2)))

            # Flatten final tensor (Final Size = (_, 2048))
            self.model.add_module("Flatten", nn.Flatten())
        

        def forward(self, x):
            return self.model(x)
        

        # Extract the features from an image passed as argument. 
        def get_features(self, image):

            # load model onto the device
            self.to(self.device)

            # Extract features
            # Always use torch.no_grad() or torch.set_grad_enabled(False) when performing inference (or during validation)
            with torch.no_grad():
                features = self(self.transform(image).to(self.device).unsqueeze(dim=0))

            # Return Normalized Features
            return normalize(features)

    model = Model()
    model.eval()

    return model

# ******************************************************************************************************************** #

def CosineDetector(image):

    # Get the models and criterion
    model = build_model()
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    # Setting up the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Extract features from the reference image
    features_1 = model.get_features(image)

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()

        # Obtain the edges
        frame = u.AUGMENT(images=np.expand_dims(frame, axis=0))[0]
        disp_frame = frame.copy()
        frame = u.preprocess(frame, change_color_space=False)

        # Extract features from the current frame
        features_2 = model.get_features(frame)

        # Calculate the Cosine Similarity between the Feature Vectors
        metric = criterion(features_1, features_2).item()

        # Add metric onto the frame
        cv2.putText(img=disp_frame, text="{:.5f}".format(metric), org=(25, 75),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        
        # Display the frame
        cv2.imshow("Feed", disp_frame)

        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Release capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
