import os
import re
import sys
import cv2
import platform
import matplotlib.pyplot as plt

import utils as u
from Models import Model

#########################################################################################################

def app():
    args_1 = "--image"
    args_2 = "--video"
    args_3 = "--realtime"
    args_4 = "--classify"
    args_5 = "--detect"
    args_6 = "--segment"
    args_7 = "--all"
    args_8 = "--model-name"
    args_9 = "--name"
    args_10 = "--downscale"

    do_downscale = None
    do_image, do_video, do_realtime = None, None, None
    do_classify, do_detect, do_segment, do_all = None, None, None, None
    name = None
    factor = None

    if args_1 in sys.argv: do_image = True
    if args_2 in sys.argv: do_video = True
    if args_3 in sys.argv: do_realtime = True
    if args_4 in sys.argv: do_classify = True
    if args_5 in sys.argv: do_detect = True
    if args_6 in sys.argv: do_segment = True
    if args_7 in sys.argv: do_all = True
    if args_8 in sys.argv: model_name = sys.argv[sys.argv.index(args_8) + 1]
    if args_9 in sys.argv: name = sys.argv[sys.argv.index(args_9) + 1]
    if args_10 in sys.argv: 
        do_downscale = True
        factor = float(sys.argv[sys.argv.index(args_10) + 1])
    
    assert(model_name is not None)
    
    if do_classify: model = Model(modeltype="classifier", model_name=model_name)
    if do_detect: model = Model(modeltype="detector", model_name=model_name)
    if do_segment: model = Model(modeltype="segmentor", model_name=model_name)
    
    model.eval()
    model.to(u.DEVICE)

    if do_image:
        assert(name is not None)
        image = cv2.imread(os.path.join(u.PATH, name))
        assert(image is not None)

        if do_classify:
            text = "Classified"
            image = u.classify(model, image)
        if do_detect:
            text = "Detected"
            if do_all is None:
                if re.match(r"f_mnet_320", model_name, re.IGNORECASE) or re.match(r"ssdlite", model_name, re.IGNORECASE):
                    image = u.detect(model, image, 320)
                elif re.match(r"ssd300", model_name, re.IGNORECASE):
                    image = u.detect(model, image, 300)
                else:
                    image = u.detect(model, image, 800)
            else:
                if re.match(r"f_mnet_320", model_name, re.IGNORECASE) or re.match(r"ssdlite", model_name, re.IGNORECASE):
                    image = u.detect(model, image, 320)
                elif re.match(r"ssd300", model_name, re.IGNORECASE):
                    image = u.detect(model, image, 300)
                else:
                    image = u.detect(model, image, 800)
        if do_segment:
            text = "Segmented"
            image = u.segment(model, image)

        plt.figure(text)
        plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))
        plt.title(text)
        plt.axis("off")
        plt.show()

    if do_video:
        assert(name is not None)
        cap = cv2.VideoCapture(os.path.join(u.PATH, name))

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                if do_downscale:
                    assert(factor is not None)
                    frame = u.downscale(frame, factor)

                if do_classify:
                    text = "Classified"
                    frame = u.classify(model, frame)
                if do_detect:
                    text = "Detected"
                    if do_all is None:
                        if re.match(r"f_mnet_320", model_name, re.IGNORECASE) or re.match(r"ssdlite", model_name, re.IGNORECASE):
                            image = u.detect(model, frame, 320)
                        elif re.match(r"ssd300", model_name, re.IGNORECASE):
                            image = u.detect(model, frame, 300)
                        else:
                            image = u.detect(model, frame, 800)
                    else:
                        if re.match(r"f_mnet_320", model_name, re.IGNORECASE) or re.match(r"ssdlite", model_name, re.IGNORECASE):
                            image = u.detect(model, frame, 320)
                        elif re.match(r"ssd300", model_name, re.IGNORECASE):
                            image = u.detect(model, frame, 300)
                        else:
                            image = u.detect(model, frame, 800)
                if do_segment:
                    text = "Segmented"
                    frame = u.segment(model, frame)
                
                cv2.imshow(text, frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                cap.set(cv2.CAP_PROP_FRAME_COUNT, 0)
        
        cap.release()
        cv2.destroyAllWindows()

    if do_realtime:
        if platform.system() != "Windows":
            cap = cv2.VideoCapture(u.device_id)
        else:
            cap = cv2.VideoCapture(u.device_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FPS, u.FPS)

        while cap.isOpened():
            _, frame = cap.read()

            if do_classify:
                text = "Classified"
                frame = u.classify(model, frame)
            if do_detect:
                text = "Detected"
                if do_all is None:
                    if re.match(r"f_mnet_320", model_name, re.IGNORECASE) or re.match(r"ssdlite", model_name, re.IGNORECASE):
                        image = u.detect(model, frame, 320)
                    elif re.match(r"ssd300", model_name, re.IGNORECASE):
                        image = u.detect(model, frame, 300)
                    else:
                        image = u.detect(model, frame, 800)
                else:
                    if re.match(r"f_mnet_320", model_name, re.IGNORECASE) or re.match(r"ssdlite", model_name, re.IGNORECASE):
                        image = u.detect(model, frame, 320)
                    elif re.match(r"ssd300", model_name, re.IGNORECASE):
                        image = u.detect(model, frame, 300)
                    else:
                        image = u.detect(model, frame, 800)
            if do_segment:
                text = "Segmented"
                frame = u.segment(model, frame)
            
            cv2.imshow(text, frame)
            if cv2.waitKey(1) == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

#########################################################################################################
