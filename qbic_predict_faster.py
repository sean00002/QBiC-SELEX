#!/usr/bin/env python3
"""Backward-compatible entry point; equivalent to `qbic-selex predict ...`.

Kept so commands written for earlier versions keep working. New code should
use the `qbic-selex` command installed by `pip install qbic-selex`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from qbic_selex.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["predict", *sys.argv[1:]]))
