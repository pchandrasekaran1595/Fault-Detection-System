import cv2
import platform
import torch
from torch import nn
from torchvision import models, transforms, ops

import utils as u

# ******************************************************************************************************************** #

# Normalize the vector to a min-max of [0, 1]
def normalize(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x

# ******************************************************************************************************************** #

# Function to build the model
def build_models():
    class FeatureExtract(nn.Module):
        def __init__(self):
            super(FeatureExtract, self).__init__()

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
            # Preprocess the image
            image = u.preprocess(image)

            # Load model onto the device
            self.to(self.device)

            # Extract features
            # Always use torch.no_grad() or torch.set_grad_enabled(False) when performing inference (or during validation)
            with torch.no_grad():
                features = self(self.transform(image).to(self.device).unsqueeze(dim=0))

            # Return Normalized Features
            return normalize(features)
    

    class RoIExtract(nn.Module):
        def __init__(self):
            super(RoIExtract, self).__init__()

            # Size expected by the model
            self.size = 320

            # Device on which to run inference on. (This is now platform aware; will choose NVIDIA GPU if present, else will run on CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Transformations on Image Data Expected by the Model
            self.transform = transforms.Compose([transforms.ToTensor(), ])

            # Get the model
            self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True)
        

        def forward(self, x):
            return self.model(x)
        
        # Extract the bounding boxes from image passed as argument
        def get_bboxes(self, image):
            h, w, _ = image.shape

             # Preprocess the image
            image = u.preprocess(image)

            # Load model onto the device
            self.to(self.device)

            with torch.no_grad():
                output = self(self.transform(image).to(self.device).unsqueeze(dim=0))[0]
            cnts, scrs = output["boxes"], output["scores"]

            # Get the bounding box if an object is detected
            if len(cnts) != 0:
                cnts = ops.clip_boxes_to_image(cnts, (self.size, self.size))
                best_index = ops.nms(cnts, scrs, 0.1)[0]
                x1, y1, x2, y2 = int(cnts[best_index][0] * (w/self.size)), int(cnts[best_index][1] * (h/self.size)), int(cnts[best_index][2] * (w/self.size)), int(cnts[best_index][3] * (w/self.size))
                return x1, y1, x2, y2
            else:
                return None, None, None, None

    fea_extractor = FeatureExtract()
    fea_extractor.eval()
    roi_extractor = RoIExtract()
    roi_extractor.eval()
    return fea_extractor, roi_extractor

# ******************************************************************************************************************** #

def CosineDetector(image, clipLimit):
    """
        image     : (np.ndarray) Image data with which realtime feeed is compared to
        clipLimit : (float) cliplimit to be used with CLAHE preprocessing
    """
    # Get the models and criterion
    fea_extractor, roi_extractor = build_models()
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    # Initialize the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Extract features from the ROI of the image
    x1, y1, x2, y2 = roi_extractor.get_bboxes(image)
    if x1 is not None:
        features_1 = fea_extractor.get_features(image[y1:y2, x1:x2])
    else:
        features_1 = fea_extractor.get_features(image)

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()
        frame = u.clahe_equ(frame, clipLimit=clipLimit)
        disp_frame = frame.copy()
        
        # Extract features from the ROI of the current frame
        x1, y1, x2, y2 = roi_extractor.get_bboxes(frame)
        if x1 is not None:
            features_2 = fea_extractor.get_features(frame[y1:y2, x1:x2])
        else:
            features_2 = fea_extractor.get_features(frame[y1:y2, x1:x2])

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
        
    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
