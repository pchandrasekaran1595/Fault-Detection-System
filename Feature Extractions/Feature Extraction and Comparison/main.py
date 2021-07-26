"""
    Entry point into the application
"""

import sys
import cli


def main():
    cli.app()


if __name__ == "__main__":
    sys.exit(main() or 0)
