"""Helhest gradient optimization — Brax positional pipeline.

Tune --lr if loss oscillates (try 0.001, 0.0001).

Usage:
    python examples/comparison_gradient/helhest/brax_positional.py
    python examples/comparison_gradient/helhest/brax_positional.py --lr 0.001
    python examples/comparison_gradient/helhest/brax_positional.py --lr 0.0001 --iters 100
    python examples/comparison_gradient/helhest/brax_positional.py --save results/brax_positional.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import brax_sim


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Adam learning rate (default: 0.001)"
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Simulated duration in seconds (default: 0.5 = T=50 steps)",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--kv", type=float, default=150.0)
    parser.add_argument("--baumgarte", type=float, default=None)
    parser.add_argument("--elasticity", type=float, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    sys.argv = [
        sys.argv[0],
        "--pipeline",
        "positional",
        "--dt",
        str(args.dt),
        "--kv",
        str(args.kv),
        "--duration",
        str(args.duration),
        "--lr",
        str(args.lr),
        "--iters",
        str(args.iters),
    ]
    if args.baumgarte is not None:
        sys.argv += ["--baumgarte", str(args.baumgarte)]
    if args.elasticity is not None:
        sys.argv += ["--elasticity", str(args.elasticity)]
    if args.save is not None:
        sys.argv += ["--save", args.save]

    brax_sim.main()


if __name__ == "__main__":
    main()
