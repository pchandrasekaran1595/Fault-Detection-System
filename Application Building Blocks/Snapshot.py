"""
    CLI Snapshot Capture
"""

import os
import cv2
import platform
import shutil
import numpy as np

import utils as u

# ******************************************************************************************************************** #

"""
    1. For the CLI version, use cv2.waitkey() to bind a button press to frame capture
    2. For the GUI version, either bind a specifig button to a callbakc or simply use callbacks.
"""

# ******************************************************************************************************************** #
