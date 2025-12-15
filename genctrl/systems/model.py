# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import copy
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from genctrl.factory.model_library import FACTORY
from genctrl.systems.t2i_generator import TextToImageGenerator
from genctrl.utils.utils import batched_generate, get_model_type_from_name


class Model(torch.nn.Module):
    def __init__(self, model):
        """
        Initialize the model class.
            :param model: model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
        """
        super(Model, self).__init__()
        self.model = model

    def __str__(self):
        return self.model.__str__()

    @classmethod
    def _from_hf_llm(cls, hf_model_name: str, cache_dir: str, **sampling_config):
        """
        Initialize the model class from a model.
            :param model: model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, cache_dir=cache_dir, force_download=False
        )
        llm = AutoModelForCausalLM.from_pretrained(
            hf_model_name, device_map="auto", cache_dir=cache_dir, force_download=False
        )

        def model(x, u, t=0):
            # Generate a string based on the input x and u
            is_template = isinstance(u[0], (dict, list))
            if is_template:
                if t == 0:
                    for i, ui in enumerate(u):
                        ui[0]["content"] = x[i] + "\n" + ui[0]["content"]

                    # Initiate the chat history with the user input
                    chats = u
                elif t > 0:
                    # Add the current user input to the chat history
                    chats = [x[i] + u[i] for i in range(len(x))]

                prompts = [
                    tokenizer.apply_chat_template(
                        chat_i,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,  # Qwen/Qwen3 models accept this
                    )
                    for chat_i in chats
                ]
            else:
                prompts = [xi + ui for xi, ui in zip(x, u)] if u is not None else x

            # Copying config so we can remove batch_size
            copy_config = copy.deepcopy(sampling_config)
            batch_size = copy_config.pop("batch_size", 32)
            x_outs = batched_generate(
                prompts,
                llm,
                tokenizer,
                copy_config,
                batch_size=batch_size,
                t=t,
                # batch_size=int(128 * (0.5) ** t), # Decrease batch size over time steps to save memory
            )

            # some templates introduce some artifacts, remove
            if is_template:
                x_outs = [x.replace("user", "").strip() for x in x_outs]
                x_outs_chat = [
                    [{"role": "assistant", "content": out_i}] for out_i in x_outs
                ]

                # Append this dict to x
                x_next = [chats[i] + x_outs_chat[i] for i in range(len(prompts))]
            else:
                x_next = [prompts[i] + x_outs[i] for i in range(len(prompts))]

            # state: the conversation trace so far (incl. prompt). out: only the new tokens generated
            return x_next, x_outs

        return cls(model)

    @classmethod
    def _from_hf_t2i(cls, hf_model_name: str, cache_dir: str, **model_config):
        """
        Initialize the model class from a model.
            :param model: model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
        """
        num_inference_steps = model_config.get("num_inference_steps", None)
        t2im = TextToImageGenerator(
            hf_model_name, cache_dir=cache_dir, num_inference_steps=num_inference_steps
        )

        def model(x, u, t=0):
            # Generate a string based on the input x and u
            prompts = [xi + str(ui) for xi, ui in zip(x, u)] if u is not None else x

            outputs = t2im.generate_images(
                prompts=prompts,
                **model_config,  # steps=4, output_dir, num_images_per_prompt=1
            )

            # state is the prompt + output; also return just the image
            return [(prompts[i], outputs[i]) for i in range(len(prompts))], outputs

        return cls(model)

    @classmethod
    def from_hf_model(cls, model_name: str, cache_dir: str, **model_config):
        """
        Initialize the model class from a model.
            :param model: model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
        """
        model_type = get_model_type_from_name(model_name)
        if model_type in ("llm", "ti2i"):
            return cls._from_hf_llm(model_name, cache_dir=cache_dir, **model_config)
        elif model_type == "t2i":
            return cls._from_hf_t2i(model_name, cache_dir=cache_dir, **model_config)

    @classmethod
    def from_callable(cls, model: callable):
        """
        Initialize the model class from a callable.
            :param model: model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
        """
        return cls(model)

    @classmethod
    def from_factory(cls, model_name: str, sampling_config: dict = {}):
        """
        Initialize the model class from a model.
            :param model: model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
        """
        if model_name not in FACTORY:
            raise ValueError(f"Model {model_name} not found in factory.")
        model = FACTORY[model_name]
        return cls(model)

    def forward(self, x: Iterable, u: Iterable, t: int = 0) -> Iterable:
        """
        Forward pass through the model. Maps the current state and input to the next state.
            :param x: Current state(s), dimensions Batch_size x x_dim
            :param u: Input, dimensions Batch_size x u_dim
            :return: Next state, dimensions Batch_size x x_dim
        """
        return self.model(x, u, t)

    def to_dict(self):
        """
        Convert the model class to a dictionary.
            :return: Dictionary representation of the model class
        """
        return {
            "model": self.model.__name__,
        }
