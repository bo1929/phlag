"""Check Phlag output against expected anomalous regions.

Usage:
    python test/check_output.py <output_file> [--start 913] [--end 1062]
"""

import argparse
import sys


def parse_output(path):
    with open(path) as f:
        lines = f.readlines()

    states = None
    posteriors = None
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            states = line.strip().split(',')
            if i + 1 < len(lines):
                posteriors = lines[i + 1].strip().split(',')
            break

    if states is None:
        print("Error: no state labels found in output", file=sys.stderr)
        sys.exit(1)

    return states, posteriors


def check(output_file, expected_start, expected_end):
    states, posteriors = parse_output(output_file)

    flagged = [j for j, s in enumerate(states) if s == '1']
    nan_count = sum(1 for s in states if s == 'nan')

    print(f"Total gene trees: {len(states)}")
    print(f"Flagged: {len(flagged)}")
    print(f"Excluded (nan): {nan_count}")

    if not flagged:
        print("No flagged regions detected.")
        return

    # Find contiguous runs
    runs = []
    start = flagged[0]
    for k in range(1, len(flagged)):
        if flagged[k] - flagged[k - 1] > 1:
            runs.append((start, flagged[k - 1]))
            start = flagged[k]
    runs.append((start, flagged[-1]))

    print(f"Detected runs: {runs}")
    print(f"Expected: [{expected_start}, {expected_end}]")

    # Check overlap with expected region
    expected = set(range(expected_start, expected_end + 1))
    detected = set(flagged)
    tp = len(expected & detected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", help="Path to Phlag output file")
    parser.add_argument("--start", type=int, default=913, help="Expected start index (default: 913)")
    parser.add_argument("--end", type=int, default=1062, help="Expected end index (default: 1062)")
    args = parser.parse_args()
    check(args.output_file, args.start, args.end)
