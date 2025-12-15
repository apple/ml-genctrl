# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

from typing import Iterable, Optional

import torch
from torch.distributions.distribution import Distribution

from genctrl.factory.model_library import FACTORY

# local imports
from genctrl.systems.model import Model


class ControlSystem:
    def __init__(
        self,
        model: callable,
        output_map: callable,
        output_space: Iterable,
        input_distribution: Optional[Distribution] = None,
        model_config: Optional[dict] = None,
    ):
        """
        Initialize the control system class.
        A model-dependent implementation of a (stochastic) control system.
            :param model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
            :param output_map: Output map that maps the current state to the output space. E.g. off-the-shelf toxicity classifier.
            :param output_space: Iterable representing the output space. E.g. [0,1] range or {0, 2, 4, 6} set of discrete categories.
            :param input_distribution: Optional distribution for permitted inputs.
            :param model_config: Optional model configuration dictionary.
        """
        self.model = model
        self.output_map = output_map
        self.input_distribution = input_distribution
        self.output_space = output_space
        self.model_config = model_config  # fill it in the from_model_name method

        # Set whether the system is stochastic
        if model_config and "do_sample" in model_config:
            self.stochastic = model_config["do_sample"]
        elif model_config and "seed" in model_config:
            # For T2I models, if the random seed is set, then the model is deterministic.
            if model_config["seed"] is not None:
                self.stochastic = False
            else:
                self.stochastic = True
        else:
            self.stochastic = True  # Assume stochastic by default (worst case)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        output_map: callable,
        output_space: Iterable,
        input_distribution: Optional[Distribution] = None,
        cache_dir: Optional[str] = None,
        model_config: Optional[dict] = None,
    ):
        """
        Initialize the control system class from a model name.
            :param model_name: Name or callable of the model
            :param output_map: (X_t) -> Y_t maps the current state to the output Y_t.
            :param output_space: Iterable representing the output space
            :param input_distribution: Optional distribution for permitted inputs
            :param cache_dir: Optional cache directory for Hugging Face models
            :param model_config: Optional dict of config for the model.
        """
        # Define the model
        if model_name in FACTORY:
            model = Model.from_factory(model_name)
            model_config = {
                "model_name": model_name,
                "do_sample": "random" in model_name,
            }
        elif callable(model_name):
            model = Model.from_callable(model_name)
            model_config = {"model_name": model_name.__name__, "do_sample": True}
        else:
            # Huggingface model
            model = Model.from_hf_model(model_name, cache_dir, **model_config)

        return cls(model, output_map, output_space, input_distribution, model_config)

    def forward(
        self,
        states: Iterable,
        inputs: Optional[Iterable] = None,
        t: Optional[int] = 0,
        **extra_variables,
    ):
        """
        Forward pass through the model.
            :param states: Current state(s), dimensions Batch_size x x_dim
            :param inputs: Input, dimensions Batch_size x u_dim
            :return: Next state, dimensions Batch_size x x_dim, and output, dimensions Batch_size x y_dim
        """
        if inputs is None:
            # extra_variables is kwargs that a feedback controller might need
            inputs = self.input_distribution.sample_n(len(states), **extra_variables)

        state_next, state_out = self.model(
            states, inputs, t
        )  # Map the current state, input to the next state and output space

        output = self.output_map(
            state_out
        )  # Map the output to the desired output space

        return state_next, output
