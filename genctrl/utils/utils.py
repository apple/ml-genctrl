# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import logging
import os
import pathlib
from functools import partial
from typing import Callable, Iterable

import requests
from omegaconf import OmegaConf
from torch.distributions.distribution import Distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from genctrl.factory.initial_states_library import INITIAL_STATES_FACTORY
from genctrl.factory.input_space_library import (
    FEEDBACK_FACTORY,
    INPUT_SPACE_FACTORY,
)
from genctrl.factory.outputs_library import (
    OUTPUT_SPACE_FACTORY,
    OUTPUTS_FACTORY,
    output_map_from_hf_detector,
    output_map_from_hf_pipeline,
    output_map_from_hf_vlm,
)

# Model type pattern mappings for inference from model names
MODEL_TYPE_PATTERNS = {
    "llm": [
        "qwen",
        "llama",
        "mistral",
        "gemma",
        "phi",
        "gpt",
        "opt",
        "bloom",
        "falcon",
        "mpt",
        "pythia",
        "stablelm",
        "vicuna",
    ],
    "t2i": ["flux", "stable-diffusion", "sdxl", "sd-"],
    "detector": ["owlv2", "owlvit", "detr", "yolo"],
}
from genctrl.factory.tasks import create_task
from genctrl.utils.distributions import (
    LLM,
    DialogueInputDistribution,
    InstructTemplateDistribution,
    StringTemplateDistribution,
    UniformDiscrete,
)


def create_task_from_config(cfg: dict) -> "Task":
    """
    Create a task from a hydra config.

    Args:
        cfg: Hydra config containing task_name, initial_states, input_space, and output_map sections

    Returns:
        Task instance with all configuration applied
    """
    # Extract task name directly from config
    task_name = OmegaConf.select(cfg, "task_name")
    if not task_name:
        # Fallback: try to extract from old-style input_space_str (for backward compatibility)
        task_name = OmegaConf.select(cfg, "input_space.input_space_str")
        if not task_name:
            raise ValueError("Config must contain 'task_name' field")

    # Keep the entire config with nested structure
    # The cfg will be stored as-is in task.config, preserving the hierarchy
    return create_task(task_name, cfg)


def get_initial_states_distribution(task: "Task") -> Distribution:
    """
    Get the initial states distribution from a task.

    Args:
        task: Task instance containing all necessary configuration

    Returns:
        Distribution over initial states
    """
    initial_states_str = task.initial_states_str

    if initial_states_str in INITIAL_STATES_FACTORY:
        initial_states = INITIAL_STATES_FACTORY[initial_states_str]
        return UniformDiscrete(initial_states)
    elif os.path.exists(initial_states_str):
        with open(initial_states_str, "r") as f:
            initial_states = f.readlines()
        initial_states = [line.strip() for line in initial_states if line.strip()]
        return UniformDiscrete(initial_states)
    elif "llm" == get_model_type_from_name(initial_states_str):
        # It is a model name for an LLM - get params from task config
        return LLM(
            model_name=initial_states_str,
            **{
                k: v
                for k, v in task.config["initial_states"].items()
                if k in ["input_prompt", "sampling_config"]
            },
        )
    else:
        # Get from task directly
        initial_states = task.get_initial_states()
        return UniformDiscrete(initial_states)


def get_output_map(task: "Task") -> tuple[Callable, Iterable]:
    """
    Get the output map from a task.

    Args:
        task: Task instance containing all necessary configuration

    Returns:
        A tuple of (output_map, output_space) that defines the output function and its range.
    """
    # Check if output_map_str is specified in the task config
    output_map_str = OmegaConf.select(task.config, "output_map.output_map_str")

    if output_map_str:
        # Check if it's a simple key in OUTPUTS_FACTORY
        if output_map_str in OUTPUTS_FACTORY:
            output_map = OUTPUTS_FACTORY[output_map_str]
            output_space = task.get_output_space()
        else:
            # It's a HuggingFace model name - determine which factory to use
            model_type = get_model_type_from_name(output_map_str)

            if model_type == "detector":
                # Use object detection model
                output_map, output_space = output_map_from_hf_detector(
                    hf_model_name=output_map_str,
                    question=OmegaConf.select(task.config, "output_map.question"),
                    bounds=task.get_output_space(),
                    threshold=OmegaConf.select(
                        task.config, "output_map.threshold", default=0.2
                    ),
                )
            elif model_type == "scorer":
                # Use text classification model
                output_map, output_space = output_map_from_hf_pipeline(
                    hf_model_name=output_map_str,
                    task="text-classification",
                    label=OmegaConf.select(
                        task.config, "output_map.label", default="formal"
                    ),
                    device=OmegaConf.select(
                        task.config, "output_map.device", default="gpu"
                    ),
                )
            elif model_type in ["ti2i", "vlm"]:
                # Use vision-language model
                output_map, output_space = output_map_from_hf_vlm(
                    hf_model_name=output_map_str,
                    question=OmegaConf.select(task.config, "output_map.question"),
                    bounds=task.get_output_space(),
                )
            else:
                raise ValueError(
                    f"Unsupported model type '{model_type}' for output_map_str: {output_map_str}"
                )
    else:
        # Fall back to task's get_output_map method
        output_map = task.get_output_map()
        output_space = task.get_output_space()

    # Handle special cases that need partial application
    question = OmegaConf.select(task.config, "output_map.question")
    if task.name == "object_position" and question is not None:
        output_map = partial(output_map, obj=question)

    return output_map, output_space


def get_input_distribution(task: "Task") -> DialogueInputDistribution:
    """
    Get the input distribution from a task.

    Args:
        task: Task instance containing all necessary configuration

    Returns:
        DialogueInputDistribution for the task
    """
    is_dialogue = task.is_dialogue

    # Get the input space from task
    input_distribution_parameters = task.get_input_space()

    # Check if instruction-tuned
    if task.instruction_tuned and not is_dialogue:
        template, arg_distributions, system_prompt = input_distribution_parameters
        initial_input_distribution = InstructTemplateDistribution(
            template, arg_distributions, system_prompt=system_prompt
        )
        return DialogueInputDistribution(
            indexer=lambda t: initial_input_distribution,
            distributions=[initial_input_distribution],
        )
    elif is_dialogue:
        # The first input should be the typical instruct one
        template, arg_distributions, system_prompt = input_distribution_parameters
        initial_input_distribution = InstructTemplateDistribution(
            template, arg_distributions, system_prompt=system_prompt
        )

        # Get feedback distribution from task
        feedback_distribution = task.get_feedback_function()

        # Put them together
        return DialogueInputDistribution(
            indexer=lambda t: (
                initial_input_distribution if t == 0 else feedback_distribution
            ),
            distributions=[initial_input_distribution, feedback_distribution],
        )
    else:
        # Standard template (e.g., for image generation)
        template, arg_distributions = input_distribution_parameters
        obj = OmegaConf.select(task.config, "input_space.obj")
        formatters = {"object": obj} if obj is not None else None
        input_distribution = StringTemplateDistribution(
            template, arg_distributions, formatters=formatters
        )
        return DialogueInputDistribution(
            indexer=lambda t: input_distribution,
            distributions=[input_distribution],
        )


def prettify_name(name: str) -> str:
    """
    So saving is less annoying with the '/' in the name
    """
    return name.split("/")[-1] if "/" in name else name


def get_model_type_from_name(model_name):
    """
    Get model type from HuggingFace API, with fallback to name-based inference.
    """
    # Try to infer from common model name patterns first (avoids API call)
    model_name_lower = model_name.lower()

    # Mock models (for testing)
    if model_name_lower.startswith("mock-"):
        # Infer type from mock model name
        if "llm" in model_name_lower:
            return "llm"
        elif "t2i" in model_name_lower or "image" in model_name_lower:
            return "t2i"
        elif "owl" in model_name_lower or "detector" in model_name_lower:
            return "detector"
        elif (
            "formality" in model_name_lower
            or "classifier" in model_name_lower
            or "scorer" in model_name_lower
        ):
            return "scorer"
        else:
            # Default mock type
            return "mock"

    # Common LLM patterns
    if any(pattern in model_name_lower for pattern in MODEL_TYPE_PATTERNS["llm"]):
        return "llm"

    # Image generation models
    if any(pattern in model_name_lower for pattern in MODEL_TYPE_PATTERNS["t2i"]):
        return "t2i"

    # Object detection models
    if any(pattern in model_name_lower for pattern in MODEL_TYPE_PATTERNS["detector"]):
        return "detector"

    # Try HuggingFace API as fallback
    try:
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        tag = data.get("pipeline_tag", "")
        if "text-generation" in tag:
            return "llm"
        elif "image-text-to-text" in tag:
            return "ti2i"
        elif "text-to-image" in tag:
            return "t2i"
        elif "text-classification" in tag:
            return "scorer"
        elif "object-detection" in tag or "zero-shot-object-detection" in tag:
            return "detector"
        else:
            raise ValueError(f"Unknown model type for model name: {model_name}.")
    except Exception as e:
        # If API call fails, raise error with helpful message
        raise ValueError(
            f"Could not determine model type for {model_name}. "
            f"API call failed: {e}. "
            f"Please ensure the model name is correct or network connection is available."
        )


def intersect_two_lists(list1, list2):
    """Intersect two sorted, disjoint lists of intervals."""
    i, j = 0, 0
    result = []
    while i < len(list1) and j < len(list2):
        a_start, a_end = list1[i]
        b_start, b_end = list2[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start <= end:  # overlap exists
            result.append([start, end])
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


def prettify_for_json(instruct_input_space):
    return [str(inp[0]) for inp in instruct_input_space]


def prettify_nested_dict_for_json(obj):
    """
    Recursively convert sets to lists so that a nested dict/list structure
    can be JSON serialized.
    """
    if isinstance(obj, dict):
        return {k: prettify_nested_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [prettify_nested_dict_for_json(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def propagate_truth_values_trajectory(stop_trajectories) -> Iterable:
    # trajectory is of the form [t=0 t=1 t=2...]
    # each t=i is a list of samples [x0 x1 x2]
    # we are making sure that if (ti, xj) = True, then (t>i, xj) = True.
    for t in range(1, len(stop_trajectories)):
        for j in range(len(stop_trajectories[t - 1])):
            if stop_trajectories[t - 1][j]:
                stop_trajectories[t][j] = True

    return stop_trajectories


def propagate_reachable_trajectory(points_reached_trajectories, stop_trajectories):
    stop_trajectories = propagate_truth_values_trajectory(stop_trajectories)

    points_reached_edited_trajectories = []
    for t, points_reached_t in enumerate(points_reached_trajectories):
        if not len(points_reached_edited_trajectories):
            # First reachable set
            points_reached_edited_trajectories.append(points_reached_t)
        else:
            points_reached_edit_t = []
            for j, stop in enumerate(stop_trajectories[t - 1]):
                if stop:
                    # The trajectory stopped
                    points_reached_edit_t.append(
                        points_reached_edited_trajectories[-1][j]
                    )
                else:
                    points_reached_edit_t.append(points_reached_t[j])
            points_reached_edited_trajectories.append(points_reached_edit_t)

    return points_reached_edited_trajectories


def save_results_summary(cfg, summary) -> pathlib.Path:
    model_name = prettify_name(cfg.model_name)
    output_map_name = prettify_name(cfg.output_map["output_map_str"])
    initial_states_name = prettify_name(cfg.initial_states["initial_states_str"])
    task_name = prettify_name(cfg.task_name)

    # Get the save directory
    fpath_str = (
        cfg.output_file
        or f"/{cfg.problem_type}_controllability_results_model_{model_name}_initial_states_{initial_states_name}_output_map_{output_map_name}_task_{task_name}.json"
    )
    save_path = pathlib.Path(cfg.output_dir) / fpath_str
    logger.info(f"Saved to: {save_path}")

    with save_path.open("w") as fp:
        json.dump(summary, fp)

    return save_path


from tqdm import tqdm


def batched_generate(
    prompts, llm, tokenizer, sampling_config, batch_size=4, t: int | None = None
):
    all_outputs = []

    # tqdm over the batches
    t_str = f" t={t + 1}" if t is not None else ""
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating{t_str}"):
        batch_prompts = prompts[i : i + batch_size]

        # tokenize
        inputs = tokenizer(
            batch_prompts, padding=True, padding_side="left", return_tensors="pt"
        ).to(llm.device)

        # forward pass
        outputs = llm.generate(
            **inputs,
            **sampling_config,
        )

        # crop out only new tokens
        out_ids = outputs[:, inputs["input_ids"].shape[1] :]

        # decode
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)

    return all_outputs
