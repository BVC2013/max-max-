"""
run_all.py
Master runner for the SVH patient-journey Python analysis.

Usage:
    python python_project/run_all.py [--data-dir data/raw] [--output-dir output]
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

# Allow imports from python_project/src/
sys.path.insert(0, str(Path(__file__).parent))

MODULES = [
    "src.data_loading",
    "src.feature_engineering",
    "src.plan_a_sdoh_funnel",
    "src.plan_b_off_hours",
    "src.plan_c_continuity",
    "src.plan_d_geography",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="SVH Patient Journey Analysis")
    parser.add_argument("--data-dir",   default="data/raw",   help="Directory containing raw CSVs")
    parser.add_argument("--output-dir", default="output",     help="Root output directory")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plot_dir   = output_dir / "plots"
    processed_dir = Path("data/processed")

    plot_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for mod_name in MODULES:
        print(f"\n{'='*60}")
        print(f"Running: {mod_name}")
        print(f"{'='*60}")
        t0 = time.time()
        mod = importlib.import_module(mod_name)
        mod.run(data_dir=data_dir, output_dir=output_dir, processed_dir=processed_dir)
        print(f"  Completed in {time.time() - t0:.1f}s")

    print(f"\nAll modules complete. Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
