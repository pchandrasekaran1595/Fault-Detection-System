import cv2
import sys
import platform
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CAM_WIDTH, CAM_HEIGHT, FPS = 640, 360, 30

# ******************************************************************************************************************** #

def app():
    args_1 = "--type"
    model_type = "MiDaS_small"

    if args_1 in sys.argv:
        model_type = sys.argv[sys.argv.index(args_1) + 1]
    
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()


    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    if platform.system() != 'Windows':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            prediction = midas(transform(frame).to(device)).squeeze()

        output = prediction.cpu().numpy()
        depth_frame = cv2.resize(src=output, dsize=(CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)

        cv2.imshow("Depth Frame", depth_frame)
        if cv2.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
