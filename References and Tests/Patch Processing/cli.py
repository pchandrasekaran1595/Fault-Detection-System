"""
    CLI Application
"""

import sys

import utils as u
from Processor import process_video_as_patches

# ******************************************************************************************************************** #

def app():
    args_1 = "--pw"
    args_2 = "--ph"
    args_3 = "--test"

    # Default CLI Arguments
    pw, ph, test = 48, 48, None

    # CLI Argument Handling
    if args_1 in sys.argv:
        pw = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        ph = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        test = int(sys.argv[sys.argv.index(args_3) + 1])
    
    u.breaker()
    u.myprint("--- Application Start ---", color="green")
    process_video_as_patches(pw, ph, test=test)
    u.myprint("\n--- Application End ---", color="green")
    u.breaker()

# ******************************************************************************************************************** #
