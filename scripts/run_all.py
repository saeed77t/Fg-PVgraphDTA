"""Python launcher for all experiment suites; options are forwarded to the CLI."""

import sys

from fgpvdta.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["suite", "--name", "all", *sys.argv[1:]]))
