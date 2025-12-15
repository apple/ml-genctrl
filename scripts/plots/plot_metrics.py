# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import argparse
import pathlib

import matplotlib.pyplot as plt

from genctrl.utils.results import parse_controllability_results, plot_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        required=True,
        type=pathlib.Path,
        help="Results json file(s).",
        nargs="+",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        required=False,
        type=pathlib.Path,
        help="Output file with png/jpg/pdf extension.",
    )
    parser.add_argument("--time-step", type=int, default=0, help="Timestep to plot.")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for file in args.json:
        xs, ys, target, info = parse_controllability_results(
            file, time_step=args.time_step
        )
        axes = plot_metrics(axes=axes, xs=xs, ys=ys, target=target, info=info)

    plt.suptitle(info["task"])
    plt.tight_layout()
    if args.outfile is not None:
        plt.savefig(args.outfile, dpi=200)
        print(f"\nFig saved as {args.outfile.absolute()}")
    else:
        plt.show()
