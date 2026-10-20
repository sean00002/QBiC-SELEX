#!/usr/bin/env python3
"""Backward-compatible entry point; equivalent to `qbic-selex model-map ...`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from qbic_selex.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["model-map", *sys.argv[1:]]))
