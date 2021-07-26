import sys
import cli
import gui


def main():
    args = "--gui"
    with_gui = False
    if args in sys.argv:
        with_gui = True
    
    if with_gui:
        gui.app()
    else:
        cli.app()


if __name__ == "__main__":
    sys.exit(main() or 0)
