"""
    Realtime Inference
"""

import cv2
import torch
import utils as u

# ******************************************************************************************************************** #

# Inference Helper
def __help__(frame=None, model=None, fea_extractor=None, show_prob=True, pt1=None, pt2=None):
    """
        frame         : Current frame being processed
        model         : Siamese Network Model
        fea_extractor : Feature Extraction Model
        show_prob     : Flag to control whether to display the similarity score
        pt1           : Start Point of the Reference Bounding Box
        pt2           : End Point of the Reference Bounding Box
    """
    disp_frame = frame.copy()

    # Center Crop + Resize
    frame = u.preprocess(frame, change_color_space=False)

    # Perform Inference on current frame
    with torch.no_grad():
        features = u.normalize(fea_extractor(u.FEA_TRANSFORM(frame).to(u.DEVICE).unsqueeze(dim=0)))
        y_pred = torch.sigmoid(model(features))[0][0].item()

    # Prediction > Upper Bound                 -----> Match
    # Lower Bound <= Prediction <= Upper Bound -----> Possible Match
    # Prediction < Lower Bound                 -----> No Match
    if show_prob:
        if y_pred >= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.CLI_GREEN, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_GREEN, thickness=2)

        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Error, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.CLI_ORANGE, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_ORANGE, thickness=2)

        else:
            cv2.putText(img=disp_frame, text="No Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.CLI_RED, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_RED, thickness=2)
    else:
        if y_pred >= u.upper_bound_confidence:
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_GREEN, thickness=2)
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_ORANGE, thickness=2)
        else:
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_RED, thickness=2)
    return disp_frame

# ******************************************************************************************************************** #


"""
    Video Handling
"""