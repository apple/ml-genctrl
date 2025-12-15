# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import warnings
from hashlib import algorithms_guaranteed

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import genctrl.utils.setup_utils as setup_utils  # wandb, yaml loading, etc.
import genctrl.utils.utils as utils

# local imports
from genctrl.systems.control_system import ControlSystem
from genctrl.utils.results import (
    LABELS,
    compute_stats,
    compute_stats_categorical,
    parse_controllability_results,
)
from genctrl.utils.setup_utils import log_wandb
from genctrl.verifiers.controllability import Controllability
from genctrl.verifiers.reachability import Reachability

# Resolver for range set in config yamls
OmegaConf.register_new_resolver(
    # "range_set", lambda start, end: list(range(int(start), int(end)))
    "range_set",
    lambda start, end: [int(start), int(end)],
)


def prompting_controllability(cfg: DictConfig):
    """
    Main function to run the controllability checks by prompting LLMs.
    """
    # Create task from config
    task = utils.create_task_from_config(cfg)
    logger.info(f"Created task: {task.name}")
    logger.info(f"  - Model name: {cfg.model_name}")
    logger.info(f"  - Instruction tuned: {task.instruction_tuned}")
    logger.info(f"  - Dialogue mode: {task.is_dialogue}")
    logger.info(f"  - Supports dialogue: {task.supports_dialogue}")
    logger.info(f"  - Num shots: {task.num_shots}")
    logger.info(f"  - Initial states source: {task.initial_states_str}")

    # Load initial states from task
    initial_states_distribution = utils.get_initial_states_distribution(task)
    logger.info(
        f"  - Actual initial states used: {initial_states_distribution.enumerate_support()}"
    )

    # Load output map from task
    output_map, output_space = utils.get_output_map(task)

    all_initial_states = initial_states_distribution.enumerate_support()
    if isinstance(all_initial_states, list):
        if len(all_initial_states) < len(output_space):
            logger.warning(
                f"Initial states {all_initial_states} are fewer than output space {output_space}. "
                "This means the controllable set is smaller than the target output space."
            )

    # Load input space from task
    input_distribution = utils.get_input_distribution(task)
    logger.info(f"Input distribution for task: {task.name}")
    logger.info(f"Output space: {output_space}")

    # Define control system
    CS = ControlSystem.from_model_name(
        cfg.model_name,
        output_map,
        output_space,
        input_distribution,
        cache_dir=cfg.model_cache,
        model_config=cfg.model_config,
    )

    # Define reachability verifier
    T = cfg.time_steps
    if T > 1 and not task.supports_dialogue:
        raise RuntimeError(
            f"Asking for T>1 (dialogue mode), but task '{task.name}' does not support dialogue. "
            f"This task has no feedback function defined. Use T=1 for this task."
        )

    verifier = Reachability.from_problem_type(
        cfg.problem_type, CS, input_distribution, **cfg.reachability_params
    )

    # # Demo: get reachable set for single initial state
    # initial_state = initial_states_distribution.sample()
    # logger.info(f"Computing reachable set for {initial_state}...")
    # reachable_set, inputs_used, points_reached = verifier.get_reachable_set(initial_state, T)
    # logger.info(f"Reachable set: {sorted(reachable_set)}")

    # Demo: get controllable set over all initial states.
    logger.info("Computing controllable set...")
    controllability_verifier = Controllability.from_reachability_problem(
        verifier, **cfg.controllability_params
    )

    (
        controllable_tube,
        reachable_tubes,
        reachable_sets,
        inputs_used_all,
        points_reached_all,
        states,
        stop_conditions,
    ) = controllability_verifier.get_controllable_set(
        initial_state_distribution=initial_states_distribution,
        T=T,
        input_distribution=input_distribution,
    )

    try:
        controllable_tube = [
            sorted(controllable_set) for controllable_set in controllable_tube
        ]
    except TypeError:
        controllable_tube = [
            sorted(controllable_set, key=lambda x: (isinstance(x, str), str(x)))
            for controllable_set in controllable_tube  # Booleans come first, strings after.
        ]

    # Save the reachable set to a file
    cfg = OmegaConf.to_container(cfg, resolve=True)
    metadata = cfg

    logger.info(f"Reachable sets: {reachable_tubes}")
    logger.info(f"Controllable set: {controllable_tube}")
    # logger.info(f"Points reached: {points_reached_all}")
    # Get inputs used in a nicer format
    # if "_it" in cfg["input_space"]["input_space_str"]:
    #     inputs_used_pretty = {
    #         k: utils.prettify_for_json(v) for k, v in inputs_used_all.items()
    #     }
    #     logger.info(f"Inputs used: {inputs_used_pretty}")

    if isinstance(all_initial_states, list):
        metadata["all_initial_states"] = all_initial_states

    save_dict = metadata
    save_dict.update(
        {
            "reachable_tubes": utils.prettify_nested_dict_for_json(reachable_tubes),
            "controllable_tube": controllable_tube,
            "reachable_sets": utils.prettify_nested_dict_for_json(reachable_sets),
            "inputs_used": inputs_used_all,
            "points_reached": points_reached_all,
            "stop_conditions": stop_conditions,
            "states": states,
        }
    )

    return save_dict


@hydra.main(
    config_path="../configs",
    config_name="llm_num_chars",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    wandb_run = setup_utils.setup_wandb(cfg)

    # Magic happens.
    summary = prompting_controllability(cfg)

    # Save
    results_file = utils.save_results_summary(cfg, summary)

    # Log to wandb if ready
    if wandb_run is not None:
        # Get metrics
        xs, ys, target, info = parse_controllability_results(results_file, time_step=-1)
        if cfg.task_name in ["even_odd", "white_bg_position_objects"]:
            metrics = [
                compute_stats_categorical(x=xi, y=yi, target=LABELS[cfg.task_name])
                for xi, yi in zip(xs.values(), ys.values())
            ]
        else:
            metrics = [
                compute_stats(x=xi, y=yi, target=target, gamma=info.get("gamma", None))
                for xi, yi in zip(xs.values(), ys.values())
            ]
        metrics_df = pd.DataFrame(metrics)
        agg_df = metrics_df.apply(np.nanmean, axis=0)
        for col in metrics_df.columns:
            agg_df[col + "_std"] = np.nanstd(metrics_df[col])
        agg = agg_df.to_dict()
        log_wandb(agg)
        wandb_run.finish()


if __name__ == "__main__":
    main()
