"""
    Entry point into the Application
"""

import sys
import gui


def main():
    gui.app()


if __name__ == "__main__":
    sys.exit(main() or 0)
