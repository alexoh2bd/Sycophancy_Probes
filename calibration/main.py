"""
CLI entrypoint for epistemic calibration pipeline.

Usage:
  python -m calibration.main extract --model_id gemma-3 --max_samples 100
  python -m calibration.main train --model_id gemma-3 --max_samples 100
  python -m calibration.main full --model_id gemma-3 --max_samples 50
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root in path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def cmd_full(args):
    """Run full pipeline: load data, extract activations, train probes."""
    from calibration.probing.train_probes import run_full_pipeline

    run_full_pipeline(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        alpha=args.alpha,
    )
    if args.plot:
        from calibration.probing.probe_heatmap import load_and_plot
        load_and_plot(args.output_dir, args.model_id)


def main():
    parser = argparse.ArgumentParser(description="Epistemic calibration pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Full pipeline
    p_full = subparsers.add_parser("full", help="Run full pipeline")
    p_full.add_argument("--model_id", default="gemma-3", help="Model ID")
    p_full.add_argument("--output_dir", default="calibration_output", help="Output directory")
    p_full.add_argument("--max_samples", type=int, default=None, help="Limit samples for debugging")
    p_full.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    p_full.add_argument("--plot", action="store_true", help="Plot R² heatmap")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
