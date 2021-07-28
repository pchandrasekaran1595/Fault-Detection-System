"""
    Entry Point into the Application
"""


import sys
import cli
import gui
import utils as u


def main():
    u.breaker()
    u.myprint("\t   --- Application Start ---", color="green")

    args = "--gui"
    with_gui = False
    if args in sys.argv:
        with_gui = True
    
    if with_gui:
        gui.app()
    else:
        cli.app()

    u.breaker()
    u.myprint("\t   --- Application End ---", color="green")
    u.breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
