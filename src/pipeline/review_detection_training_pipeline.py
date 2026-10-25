"""Run candidate training; production promotion is explicit and quality-gated."""

import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.constants import RAW_DATA_PATH
from src.pipeline.run_v2_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=RAW_DATA_PATH)
    parser.add_argument(
        "--output-dir", default="artifacts/experiments/manual-candidate"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--promote", action="store_true")
    arguments = parser.parse_args()
    run_experiment(
        arguments.dataset,
        arguments.output_dir,
        random_seed=arguments.seed,
        promote=arguments.promote,
    )


if __name__ == "__main__":
    main()
