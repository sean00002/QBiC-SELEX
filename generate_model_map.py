#!/usr/bin/env python3
"""Generate a model-to-ID mapping file from a model list.

Reads a model list file (one model path per line), extracts model names,
sorts them alphabetically, and assigns integer IDs (0, 1, 2, ...).
The output CSV can be used with qbic_predict.py --short-output --model-map.

Usage:
    python generate_model_map.py -m models.txt -o model_map.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def extract_model_name(file_path: str) -> str:
    """Extract model name from file path using first part before '.'"""
    return Path(file_path).name.split('.')[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a model-to-ID mapping file from a model list"
    )
    parser.add_argument("-m", "--models", required=True,
                       help="Text file with model paths (one per line)")
    parser.add_argument("-o", "--output", required=True,
                       help="Output mapping CSV file")
    args = parser.parse_args()

    # Read model paths
    model_names = []
    with open(args.models, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                model_names.append(extract_model_name(line))

    if not model_names:
        print("Error: No model paths found in input file", file=sys.stderr)
        sys.exit(1)

    # Sort alphabetically and assign IDs
    model_names_sorted = sorted(set(model_names))

    # Write mapping CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'model_name'])
        for model_id, model_name in enumerate(model_names_sorted):
            writer.writerow([model_id, model_name])

    print(f"Generated mapping for {len(model_names_sorted)} models: {args.output}")


if __name__ == '__main__':
    main()
