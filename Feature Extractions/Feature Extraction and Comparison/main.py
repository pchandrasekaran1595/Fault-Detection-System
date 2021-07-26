"""
    Entry point into the application
"""

import sys
import cli
import utils as u


def main():
    u.breaker()
    u.myprint("\t   --- Application Start ---", color="green")

    cli.app()
    
    u.breaker()
    u.myprint("\t   --- Application End ---", color="green")
    u.breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
