"""Helhest gradient optimization — Brax generalized pipeline.

Stable at dt=0.002 (full 3s trajectory). Each iteration takes ~75s (T=1500 steps).
For quick testing use --duration 0.3 (T=150 steps, ~8s/iter).
Tune --lr if loss diverges (try 0.0001).

Usage:
    python examples/comparison_gradient/helhest/brax_generalized.py
    python examples/comparison_gradient/helhest/brax_generalized.py --duration 0.3
    python examples/comparison_gradient/helhest/brax_generalized.py --lr 0.0001
    python examples/comparison_gradient/helhest/brax_generalized.py --save results/brax_generalized.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import brax_sim

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr",         type=float, default=0.001,
                        help="Adam learning rate (default: 0.001)")
    parser.add_argument("--iters",      type=int,   default=50)
    parser.add_argument("--duration",   type=float, default=3.0,
                        help="Simulated duration in seconds (default: 3.0 = T=1500 steps, ~75s/iter). "
                             "Use 0.3 for fast testing.")
    parser.add_argument("--dt",         type=float, default=0.002)
    parser.add_argument("--kv",         type=float, default=150.0)
    parser.add_argument("--baumgarte",  type=float, default=None)
    parser.add_argument("--save",       type=str,   default=None)
    args = parser.parse_args()

    sys.argv = [sys.argv[0],
        "--pipeline", "generalized",
        "--dt",       str(args.dt),
        "--kv",       str(args.kv),
        "--duration", str(args.duration),
        "--lr",       str(args.lr),
        "--iters",    str(args.iters),
    ]
    if args.baumgarte is not None: sys.argv += ["--baumgarte", str(args.baumgarte)]
    if args.save      is not None: sys.argv += ["--save",      args.save]

    brax_sim.main()

if __name__ == "__main__":
    main()
