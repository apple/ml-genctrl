# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import pathlib
import shutil

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tree

from genctrl.factory.input_space_library import parse_input_strings
from genctrl.verifiers.reachability import QuantizedReachability


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def setup_latex_for_plots():
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times", "DejaVu Serif", "Computer Modern Roman"],
            "font.weight": "normal",
            "text.usetex": True,
        }
    )

    # Function to check if a command exists
    def command_exists(cmd):
        return shutil.which(cmd) is not None

    # Check for 'latex' or 'pdflateatex'
    # Matplotlib's texmanager often defaults to 'latex' but might use 'pdflatex'
    # Let's check for 'pdflatex' as it's commonly used for modern LaTeX documents
    if not command_exists("pdflatex") and not command_exists("latex"):
        print("LaTeX not found. Disabling 'text.usetex' in Matplotlib.")
        plt.rcParams["text.usetex"] = False
    else:
        # You can optionally set it to True if you want to ensure it's on when available
        # and it hasn't been set elsewhere.
        # By default, 'text.usetex' is False, so this 'else' block might not be strictly necessary
        # unless you explicitly want to enable it here.
        print("LaTeX found. 'text.usetex' can be used if desired.")
        # mpl.rcParams['text.usetex'] = True # Uncomment if you want to force enable when found


setup_latex_for_plots()


COLORS = {
    "google/gemma-3-1b-it": "gray",
    "google/gemma-3-4b-it": "blue",
    "Qwen/Qwen3-4B": "green",
    "black-forest-labs/FLUX.1-dev": "blue",
    "black-forest-labs/FLUX.1-schnell": "green",
    "stabilityai/stable-diffusion-xl-base-1.0": "orange",
    "tianweiy/DMD2": "purple",
}

MARKERS = {
    "google/gemma-3-1b-it": "^",
    "google/gemma-3-4b-it": "d",
    "Qwen/Qwen3-4B": "o",
    "black-forest-labs/FLUX.1-dev": "d",
    "black-forest-labs/FLUX.1-schnell": "o",
    "stabilityai/stable-diffusion-xl-base-1.0": "s",
    "tianweiy/DMD2": "^",
}

MODEL_PRETTY = {
    "google/gemma-3-1b-it": "Gemma-3-1B",
    "google/gemma-3-4b-it": "Gemma-3-4B",
    "Qwen/Qwen3-4B": "Qwen-3-4B",
    "black-forest-labs/FLUX.1-dev": "FLUX-d",
    "black-forest-labs/FLUX.1-schnell": "FLUX-s",
    "stabilityai/stable-diffusion-xl-base-1.0": "SDXL",
    "tianweiy/DMD2": "DMD2",
}


METRIC_PRETTY = {
    "spearman": r"$\rho \;(\uparrow)$",
    "pearson": r"$R \;(\uparrow)$",
    "mae": r"$\mathrm{MAE} \;(\downarrow)$",
    # "coverage": r"\mathrm{cvg} \;(\uparrow)",
    "coverage": r"Controllable set $\mathcal{{C}}_{{t={time_step:d}}} \;(\uparrow)$",
    "accuracy": "Acc",
    "f1": "F1",
}

LABELS = {
    "even_odd": ["True", "False", "error"],
    "white_bg_position_objects": [
        "top left",
        "top right",
        "bottom left",
        "bottom right",
        "center",
        "none",
    ],
}

PRETTY_LABELS = {
    "even_odd": [r"\texttt{E}", r"\texttt{O}", r"\texttt{-}"],
    "white_bg_position_objects": [
        r"\texttt{TL}",
        r"\texttt{TR}",
        r"\texttt{BL}",
        r"\texttt{BR}",
        r"\texttt{C}",
        r"\texttt{-}",
    ],
}


def parse_controllability_results(
    json_file: pathlib.Path, time_step: int = None
) -> tuple[dict, dict, set | list, dict]:
    with json_file.open() as f:
        results = json.load(f)
    task = results["task_name"]
    model = results["model_name"]
    ys = results["points_reached"]
    xs = parse_input_strings(task=task, inputs=results["inputs_used"])
    xs_flatten = np.array([v for v in xs.values()]).flatten()
    if results["problem_type"] == "discrete":
        target = set(xs_flatten.tolist())
    else:
        # Convert possible None to np.nan
        xs_flatten = xs_flatten.astype(float)
        target = (np.nanmin(xs_flatten), np.nanmax(xs_flatten))

    if time_step is not None:
        xs = {k: v[0] for k, v in xs.items()}
        first_y = ys[list(ys.keys())[0]]
        assert time_step < len(first_y), (
            f"timestep {time_step} is too large. Only timesteps {list(range(len(first_y)))} are available."
        )
        ys = {k: v[time_step] for k, v in ys.items()}

    if "even_odd" in task or "white_bg_position_objects" in task:
        xs = tree.map_structure(str, xs)
        ys = tree.map_structure(str, ys)
        target = tree.map_structure(str, target)

    info = {
        "model": model,
        "task": task,
        "T": results["time_steps"],
        "time_step": time_step,
    }
    if results["problem_type"] != "discrete":
        info["gamma"] = results["reachability_params"]["gamma"]

    if "formality" in task:
        xs = tree.map_structure(lambda x: x / 100 if x is not None else x, xs)
        target = tree.map_structure(lambda x: x / 100 if x is not None else x, target)
        # info["gamma"] *= 100
    return xs, ys, target, info


def compute_stats(x, y, target: set | list, gamma: float | None) -> dict:
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()
    spearman = scipy.stats.spearmanr(x, y)[0]
    pearson = scipy.stats.pearsonr(x, y)[0]
    mae = np.mean(np.abs(x - y))
    if gamma is None:
        coverage = len(set(y).intersection(set(target))) / len(target)
    else:
        assert len(target) == 2
        gamma_cover = QuantizedReachability.compute_gamma_cover(
            points_reached=y, gamma=gamma
        )

        def trim(ab):
            a, b = ab
            a = max(min(a, target[1]), target[0])
            b = max(min(b, target[1]), target[0])
            return a, b

        gamma_cover = [trim(ab) for ab in gamma_cover]
        lengths = [g[1] - g[0] for g in gamma_cover]
        coverage = sum(lengths) / float(max(target) - min(target))

    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "coverage": coverage,
    }


def compute_stats_categorical(x, y, target: list[str] | None = None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    def make_numeric(v):
        return [target.index(vi) for vi in v]

    xnum = make_numeric(x)
    ynum = make_numeric(y)

    accuracy = accuracy_score(xnum, ynum)
    f1 = f1_score(xnum, ynum, average="macro")
    coverage = len(set(y).intersection(set(target))) / len(target)
    return {"accuracy": accuracy, "f1": f1, "coverage": coverage}


def plot_metrics(
    axes: list[plt.Axes],
    xs: dict,
    ys: dict,
    target: any,
    info: dict,
):
    assert len(axes) == 5, (
        f"Must pass 5 axes, create with 'fig, axes = plt.subplots(1, 5, figsize=(15, 3))'"
    )

    model = info["model"]
    print(f"Plotting {model}.")

    xs_all, ys_all = [], []
    for xv, yv in zip(xs.values(), ys.values()):
        xs_all.append(xv)
        ys_all.append(yv)
    xs_all = np.asarray(xs_all)
    ys_all = np.asarray(ys_all)
    assert ys_all.ndim in [1, 2], (
        f"Passed ys of dimension > 2. This probably means you passed a dialogue, which is not allowed in plot_metrics(). Please pass only 1 timestep values."
    )

    # if ys_all.ndim == 2:
    #     median = np.nanmedian(ys_all, axis=0)
    #     q25 = np.nanquantile(ys_all, axis=0, q=0.25)
    #     q75 = np.nanquantile(ys_all, axis=0, q=0.75)
    # else:
    #     median = ys_all
    #     q25 = None
    #     q75 = None

    base_task = info["task"]
    if base_task in ["even_odd", "white_bg_position_objects"]:
        metrics = [
            compute_stats_categorical(x=xi, y=yi, target=LABELS[base_task])
            for xi, yi in zip(xs.values(), ys.values())
        ]
    else:
        metrics = [
            compute_stats(x=xi, y=yi, target=target, gamma=info.get("gamma", None))
            for xi, yi in zip(xs.values(), ys.values())
        ]

    metrics_df = pd.DataFrame(metrics)

    # Start plotting
    plt.sca(axes[0])
    if base_task in ["even_odd", "white_bg_position_objects"]:
        true_y = LABELS[base_task]
        sep = 0.13
        ylabel, ycount = np.unique(ys_all.flatten(), return_counts=True)

        bars = []
        for label in true_y:
            idx = ylabel.tolist().index(label) if label in ylabel else None
            bars.append(ycount[idx] if idx is not None else 0)

        plt.bar(
            x=np.arange(len(true_y) - 1) * sep,
            height=bars[:-1],
            width=0.09,
            color=COLORS.get(model, "blue"),
        )
        plt.bar(
            x=(len(true_y) - 1) * sep,
            height=bars[-1],
            width=0.07,
            color=COLORS.get(model, "gray"),
        )

        minor = np.arange(len(true_y)) * sep
        plt.xticks(
            minor, PRETTY_LABELS[base_task], rotation=90, fontsize=7, fontweight="bold"
        )
        plt.xlabel("Measurement space $\mathcal{Y}$")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.2)
    else:
        plt.scatter(
            xs_all.flatten(),
            ys_all.flatten(),
            label=info["model"],
            color=COLORS.get(model, "blue"),
            marker=MARKERS.get(model, "o"),
            alpha=0.5,
            s=3,
        )
        # if q25 is not None:
        #     plt.fill_between(
        #         x=xs_unique, y1=q25, y2=q75, color=COLORS.get(model, "blue"), alpha=0.2
        #     )
        plt.xlabel("Input")
        plt.ylabel("Output")
        # ↓ Keep around 4 ticks automatically
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        plt.grid(visible=True, alpha=0.2)

    # metrics_names = ["spearman", "pearson", "mae", "coverage"] if "spearman" in metrics else ["accuracy", "f1", "coverage"]
    for i, metric in enumerate(metrics_df.columns):
        j = 0
        plt.sca(axes[i + 1])
        v_parts = plt.violinplot(
            positions=[j],
            dataset=metrics_df[metric].dropna().values,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        w = 0.2
        m = np.nanmedian(metrics_df[metric].dropna().values)
        plt.plot((j - w, j + w), (m, m), marker=None, color=COLORS.get(model, "blue"))
        plt.scatter(
            j, m, marker=MARKERS.get(model, "o"), color=COLORS.get(model, "blue"), s=50
        )

        # Set violin parts colors
        v_parts["bodies"][0].set_facecolor(COLORS.get(model, "blue"))
        v_parts["bodies"][0].set_edgecolor(None)
        v_parts["bodies"][0].set_alpha(0.5)  # transparency
        v_parts["cmedians"].set_edgecolor(COLORS.get(model, "blue"))
        v_parts["cmedians"].set_alpha(1.0)
        v_parts["cmedians"].set_linewidth(1)
        plt.ylabel(
            rf"{METRIC_PRETTY[metric].format_map(SafeDict(time_step=info.get('time_step', 1)))}"
        )
        # plt.xticks([], [])
        j += 1
        if metric == "mae":
            plt.ylim([0, 1])
        if metric == ("coverage"):
            plt.ylim([0, 1])
        if metric == "spearman" or metric == "pearson":
            plt.ylim([0, 1])
        plt.grid(True, alpha=0.2)

    legend_handles = []

    ax = plt.gca()
    if ax.get_legend():
        legend_handles = ax.get_legend().legend_handles

    legend_handles.append(
        mlines.Line2D(
            [0],
            [0],
            color=COLORS.get(model, "blue"),
            linestyle="-",
            marker=MARKERS.get(model, "o"),
            markersize=8,
            label=MODEL_PRETTY.get(model, model),
        )
    )

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        borderaxespad=0.0,
        frameon=False,
        handlelength=1,
    )
    return axes
