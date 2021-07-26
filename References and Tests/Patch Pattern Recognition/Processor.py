import cv2
import platform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL

import utils as u
from Models import build_model

# ******************************************************************************************************************** #

# Dataset Template 
class DS(Dataset):
    def __init__(self, X=None, transform=None):
        self.transform = transform
        self.X = X

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.transform(self.X[idx])

# ******************************************************************************************************************** #

def process_patches_in_video(patch, similarity):
    if patch is None:
        u.myprint("\nError Reading Patch File", "red")
        return
    
    # Get the model
    model = build_model()

    # Infer the patch height and width from the patch
    ph, pw, _ = patch.shape

    # Extract the features from the patch
    patch_features = model.get_features(patch)
    
    # Setting up capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    print("")
    # Read data from the capture object
    while cap.isOpened():
        _, frame = cap.read()
        h, w, _ = frame.shape

        # Calculate the number of rows and columns that will be present in the patched image
        if h % ph == 0 : num_cols = int(h/ph)
        else : num_cols = int(h/ph) + 1
        if w % pw == 0 : num_rows = int(w/pw)
        else : num_rows = int(w/pw) + 1

        # Resize frame to avoid any errors
        frame = cv2.resize(src=frame, dsize=(num_rows*pw, num_cols*ph), interpolation=cv2.INTER_AREA)
        disp_frame = frame.copy()

        # patches: Holds the image data of the patches
        # patches_idx: Holds the array indexes at which patch was made
        patches = []
        patches_ = []
        patches_idx = []
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                patches_idx.append([i, ph+i, j, pw+j])
                patches_.append(frame[i:ph+i, j:pw+j, :])
                patches.append(u.preprocess(frame[i:ph+i, j:pw+j, :]))
            
        # Setup the Dataloader
        data_setup = DS(X=patches, transform=model.transform)
        data = DL(data_setup, batch_size=64, shuffle=False)
        
        # Get features of all the patches
        features = model.get_batch_features(data)
        
        # Obtain the Cosine Similarity between the reference patch Feature Vector and all the patches within the frame
        cos_sim = []
        for feat in features:
            cos_sim.append(model.get_cosine_similarity(patch_features, feat.reshape(1, -1)))
        
        # Adjust color of the patch in accordance with its Cosine Similarity metric
        for i in range(num_cols):
            for j in range(num_rows):
                disp_frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = patches_[i * num_rows + j]
                if cos_sim[i * num_rows + j] > similarity:
                    disp_frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw, 1] = 200

        # Display the frame
        cv2.imshow("Pattern Processed Frame", disp_frame)

        # Press 'q' to Quit
        if cv2.waitKey(u.WAIT_DELAY) == ord("q"):
            break
    
    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
