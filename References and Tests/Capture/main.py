"""
    Entry point into the Application.
    
    Simple script to capture images from the webcam. Specify --nogui if gui is not needed.
"""

import sys
import cli
import gui

# ******************************************************************************************************************** #

def main():
    args = "--nogui"

    do_gui = True
    if args in sys.argv:
        do_gui = False
    
    if do_gui:
        gui.app()
    else:
        cli.app()

# ******************************************************************************************************************** #

if __name__ == "__main__":
    sys.exit(main() or 0)
