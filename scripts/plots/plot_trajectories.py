# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import argparse
import json
import pathlib
import random
import shutil

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from black.trans import defaultdict
from matplotlib.lines import Line2D

from genctrl.factory.input_space_library import parse_input_strings
from genctrl.utils.results import (
    LABELS,
    MODEL_PRETTY,
    compute_stats,
    compute_stats_categorical,
    parse_controllability_results,
    setup_latex_for_plots,
)

setup_latex_for_plots()
cmap = plt.get_cmap("jet")

prettyprint = defaultdict(lambda: None)
prettyprint.update(
    {
        "formality_it_dialogue": "0-shot + Instruct",
        "formality_it_1shot_dialogue": "1-shot + Instruct",
        "formality_it_5shot_dialogue": "5-shot + Instruct",
        "formality": "Text formality",
        "even_odd": "Even/odd integer",
        "even_odd_it_dialogue": "0-shot + Instruct",
        "even_odd_it_1shot_dialogue": "1-shot + Instruct",
        "even_odd_it_5shot_dialogue": "5-shot + Instruct",
        "num_chars": "String length",
        "num_chars_it_dialogue": "0-shot + Instruct",
        "num_chars_it_1shot_dialogue": "1-shot + Instruct",
        "num_chars_it_5shot_dialogue": "5-shot + Instruct",
        "average_word_length": "Avg word length",
        "average_word_length_it_dialogue": "0-shot + Instruct",
        "average_word_length_it_1shot_dialogue": "1-shot + Instruct",
        "average_word_length_it_5shot_dialogue": "5-shot + Instruct",
        "saturation": "Saturation",
        "white_bg_objects": "N Objects",
        "white_bg_position_objects": "Object Position",
    }
)
prettyprint.update(MODEL_PRETTY)

discrete_map_even = {True: -1, False: 0, "error": 1}
discrete_map_pos = {
    "top left": 0,
    "top right": 1,
    "bottom left": 2,
    "bottom_right": 3,
    "center": 4,
    "error": 5,
}


def get_start_end(task: str, result: dict) -> tuple:
    if task in ["num_chars", "average_word_length", "white_bg_objects"]:
        return result["input_space"]["start"], result["input_space"]["end"]
    if task in ["formality", "saturation"]:
        return 0, 1
    if task in ["even_odd"]:
        return -1.5, 0.5
    if task in ["white_bg_position_objects"]:
        return -1.5, 4.5


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
    args = parser.parse_args()
    results = []
    for fpath in args.json:
        with open(fpath, "r") as f:
            results.append(json.load(f))

    # Check the results contain trajectories
    has_traj = all([r["time_steps"] > 1 for r in results])
    if not has_traj:
        raise RuntimeError("Results passed with T=1, no trajectories to plot.")

    ## Plot the controllable set over time
    # %%
    # VARIABLES TO PLOT
    MODELS = [ri["model_name"] for ri in results]
    PROBLEM_TYPE = results[0]["problem_type"]
    OUTPUT = results[0]["output_map"]["output_map_str"]
    TASK = results[0]["task_name"]
    # Get num_shots from initial_states if available (for LLM tasks)
    print(results[0]["initial_states"])
    num_shots_list = [ri["initial_states"]["num_shots"] for ri in results]
    SHOTS = set([f"{n}-shot" if n is not None else "0-shot" for n in num_shots_list])
    if len(SHOTS) > 1:
        raise RuntimeError("Mixing different shot counts is not allowed!")
    else:
        SHOTS = SHOTS.pop()

    trajectories_all = []
    for result in results:
        trajectories = result["points_reached"]

        # Edit to make them stop if condition met
        for x0 in result["stop_conditions"]:
            for t in range(1, len(trajectories[x0])):
                for i in range(len(trajectories[x0][t])):
                    stop = result["stop_conditions"][x0][t - 1][i]
                    trajectories[x0][t][i] = (
                        trajectories[x0][t - 1][i] if stop else trajectories[x0][t][i]
                    )

        trajectories_all.append(trajectories)
    # %%
    controllable_tubes = [
        result["controllable_tube"] for result in results
    ]  # ordered as MODELS
    # %%
    # inputs_used_all = [
    #     {x0: result["inputs_used"][x0][0] for x0 in result["inputs_used"]}
    #     for result in results
    # ]
    # inputs_used_all = [
    #     {x0: [inp[0]["content"] for inp in inputs_used[x0]] for x0 in inputs_used}
    #     for inputs_used in inputs_used_all
    # ]
    # # %%
    # # Different for every task.
    # inputs_used = parse_input_strings(task=TASK, inputs=results[0]["inputs_used"])
    # %%
    pmin = results[0]["reachability_params"]["p_min"]
    gamma = results[0]["reachability_params"]["gamma"]
    alpha = 0.9
    delta = 0.05
    eps = 0.05
    n = len(results[-1]["points_reached"]) * len(
        list(results[-1]["points_reached"].values())[-1][-1]
    )
    # %%

    # %%
    ymin, ymax = get_start_end(task=TASK, result=results[0])
    n_models = len(MODELS)
    fig = plt.figure(figsize=(4 * n_models + 3, 2.5))
    fig, axes = plt.subplots(
        figsize=(4 * n_models + 3, 2.5), ncols=n_models, sharey=True
    )
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Outer: left block for models, right skinny column for input dist (tight gap)
    outer_gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[4 * n_models, 1],  # ≈ [4]*n_models + [1]
        wspace=0.02,  # tight between last model and dist panel
    )

    # Inner: models laid out horizontally with space between them
    models_gs = outer_gs[0].subgridspec(1, n_models, wspace=0.15)
    for m, MODEL in enumerate(MODELS):
        ax = axes[m]

        # --- Uncontrollable background ---
        ax.fill_between(
            [1, len(controllable_tubes[m])],
            ymin,
            ymax,
            color="lightgray",
            hatch="///",
            edgecolor="white",
            alpha=0.5,
        )

        # --- Controllable intervals ---
        for t in range(1, len(controllable_tubes[m])):
            for interval in controllable_tubes[m][t]:
                # If quantized
                if PROBLEM_TYPE == "quantized":
                    lower, upper = interval
                elif type(interval) in (int, float) and type(interval) != bool:
                    lower, upper = interval - 0.5, interval + 0.5
                elif TASK == "even_odd":
                    lower, upper = (
                        discrete_map_even[interval] - 0.5,
                        discrete_map_even[interval] + 0.5,
                    )
                elif TASK == "white_bg_position_objects":
                    lower, upper = -0.5, 5.5

                t_start = t
                t_end = t + 1
                ax.fill_between([t_start, t_end], lower, upper, color="white", alpha=1)
                ax.fill_between(
                    [t_start, t_end], lower, upper, color="yellow", alpha=0.2
                )

        # --- Trajectories ---
        original_init_states = list(trajectories_all[m].keys())
        init_states = list(trajectories_all[m].keys())
        x0 = init_states[0]
        n_trajs = len(trajectories_all[m][x0][0])
        print(f"num states {len(init_states)} for model {m}")
        print(f"num trajs {n_trajs} for model {m}")
        for i in range(n_trajs):
            # Too many trajectories, plot 1/3 of them...
            if PROBLEM_TYPE == "quantized":
                random.shuffle(init_states)
                rng = random.random()
                # if rng > 1/15: continue

            for x0 in init_states:
                x0_traj = trajectories_all[m][x0]
                times = range(1, len(x0_traj))
                j = original_init_states.index(x0)
                traj = [x0_traj[t][i] for t in times]

                a = 0.1 if PROBLEM_TYPE == "quantized" else 0.5
                thickness = 0.5 if PROBLEM_TYPE == "quantized" else 0.7

                if TASK in ["even_odd", "white_bg_position_objects"]:
                    traj = [discrete_map_even[value] for value in traj]

                ax.plot(
                    times,
                    traj,
                    alpha=a,
                    c=f"C{j}",
                    linewidth=thickness,
                    marker="o",
                    markersize=2.0,
                )

        # --- Axis styling ---

        ax.set_title(f"{prettyprint[MODEL]} ({SHOTS})", fontsize=16)

        ax.set_xlabel(r"Dialogue turn ($t$)", fontsize=16)
        if m == 0:
            ax.set_ylabel(rf"{prettyprint[TASK]} ($y_t$)", fontsize=16)

        ax.set_ylim([ymin, ymax])
        if TASK == "num_chars":
            # # Bin centers: integers 1..10 → labels go here
            # y_centers = range(ymin, ymax + 1)
            # ax.set_yticks(y_centers)
            # ax.set_yticklabels(y_centers)

            # Bin edges: 0.5..10.5 → grid lines only here
            y_edges = [i + 0.5 for i in range(ymin, ymax + 1)]
            for y in y_edges:
                ax.axhline(y, color="lightgray", linestyle="-", linewidth=0.8, zorder=9)
        elif TASK == "even_odd":
            y_centers = [-1, 0]
            ax.set_yticks(y_centers)
            ax.set_yticklabels(["even", "odd"])

            y_edges = [i + 0.5 for i in range(-2, 2)]
            for y in y_edges:
                ax.axhline(y, color="lightgray", linestyle="-", linewidth=0.8, zorder=9)

        ax.set_xlim([1, len(controllable_tubes[m])])
        ax.set_xticks(range(1, len(controllable_tubes[m]) + 1))
        ax.tick_params(axis="both", labelsize=13)
        if PROBLEM_TYPE == "quantized":
            ax.grid()

        if m == len(MODELS) - 1:
            traj_line = Line2D(
                [], [], color="gray", linewidth=0.5, alpha=1, label=rf"Trajectories"
            )
            n_line = Line2D(
                [], [], color="gray", linewidth=0.5, alpha=0, label=rf"$N={n}$"
            )
            n_line2 = Line2D(
                [],
                [],
                color="gray",
                linewidth=0.5,
                alpha=0,
                label=rf"$k={len(init_states)}$ $m={n_trajs}$",
            )
            controllable_patch = mpatches.Patch(
                facecolor="yellow",
                alpha=0.9,
                edgecolor="black",
                linewidth=1,
                label="Controllable set",
            )

            params = mpatches.Patch(
                facecolor="white",
                alpha=0.0,
                edgecolor="none",
                label=rf"confidence $\delta={delta:.2f}$",
            )

            # Make the patch normally (black edge, hatch visible)
            uncontrollable_patch = mpatches.Patch(
                facecolor="lightgray",
                hatch="///",
                alpha=0.5,
                edgecolor="black",  # <- gives the black outline
                linewidth=1,
                label="Not controllable",
            )

            # After creating the legend, recolor hatch lines to white
            leg = ax.legend(
                handles=[
                    traj_line,
                    n_line,
                    n_line2,
                    controllable_patch,
                    params,
                    uncontrollable_patch,
                ],
                fontsize=12,
                frameon=False,
                bbox_to_anchor=(1.05, 0.5),
                loc="center left",
            )

            # Loop through legend patches and recolor their hatches
            # --- recolor hatches ---
            for patch in leg.get_patches():
                if patch.get_hatch():
                    white = mcolors.to_rgba("white")
                    patch._hatch_color = white
                    patch._hatch_edgecolor = white

    plt.tight_layout()
    outfile = args.outfile or pathlib.Path(f"trajectories_{TASK}.png")
    fig.savefig(outfile, dpi=300)
    print(f"\nFigure saved as {outfile}")

    all_metrics = []
    for fpath in args.json:
        xs, ys, target, info = parse_controllability_results(fpath)
        xs_init = {k: v[0] for k, v in xs.items()}
        if info["T"] == 1:
            ys_all = {k: v[0] for k, v in ys.items()}
        else:
            ys_all = {
                k: v[1:] for k, v in ys.items()
            }  # First contains None in dialogue setting

        for k in xs_init.keys():
            for i, yi in enumerate(ys_all[k]):
                if TASK in ["even_odd", "white_bg_position_objects"]:
                    res = compute_stats_categorical(
                        x=xs_init[k], y=yi, target=LABELS[TASK]
                    )
                else:
                    res = compute_stats(
                        x=xs_init[k], y=yi, target=target, gamma=info.get("gamma", None)
                    )
                res.update({"model": info["model"], "initial_state": k, "t": i + 1})
                all_metrics.append(res)
    metrics_df = pd.DataFrame(all_metrics)

    csv_file = outfile.with_suffix(".csv")
    metrics_df.to_csv(csv_file, index=False)
    print("\n")
    print(metrics_df)
    print(f"\nMetrics saved as {csv_file}")
