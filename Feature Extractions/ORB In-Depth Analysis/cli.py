"""
    CLI Arguments
        1. --filename  : Name of the file from which to extract ORB Features from
        2. --realtime  : Flag that controls entry into realtime analysis
        3. --nfeatures : Number of features to be used in ORB initializtion (Default: 500)
        4. --K         : Specify if you want to use Top-K Features
        5. --track     : Flag that controls entry into keypoint tracking

    Runs by default in image analysis mode
"""

import sys

import utils as u
from Analysis import image_analysis, realtime_analysis
from Track import basic_tracker

# ******************************************************************************************************************** #

def app():
    u.breaker()
    u.myprint("--- Application Start ---", color="green")

    args_1 = "--filename"
    args_2 = "--realtime"
    args_3 = "--nfeatures"
    args_4 = "--K"
    args_5 = "--track"

    # Default CLI Argument Values
    name = "Snapshot_1.png"
    do_realtime, do_analysis = None, True
    nfeatures = 500
    K = None

    if args_1 in sys.argv:
        name = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv:
        do_realtime = True
    if args_3 in sys.argv:
        nfeatures = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        K = int(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv:
        do_analysis = False

    # If --track is not specified.
    if do_analysis:
        if do_realtime is None:
            image_analysis(name, nfeatures, K=K)
        else:
            realtime_analysis(nfeatures, K=K)
    else:
        basic_tracker(nfeatures, K=K)
       

    u.myprint("\n--- Application End ---", color="green")
    u.breaker()

# ******************************************************************************************************************** #
