"""
    Entry Point into the Application.
    
    Simple script to capture images from the webcam. Specify --nogui if gui is not needed.
"""

import sys
import cli
import gui

# ******************************************************************************************************************** #

def main():
    cli.breaker()
    cli.myprint("\t   --- Application Start ---", color="green")

    args = "--nogui"

    do_gui = True
    if args in sys.argv:
        do_gui = False
    
    if do_gui:
        gui.app()
    else:
        cli.app()
    
    cli.breaker()
    cli.myprint("\t   --- Application End ---", color="green")
    cli.breaker()

# ******************************************************************************************************************** #

if __name__ == "__main__":
    sys.exit(main() or 0)
