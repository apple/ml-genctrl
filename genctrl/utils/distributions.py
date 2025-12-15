# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import copy
import itertools
import logging
from typing import Any, Iterable

import inflect
import torch
from torch.distributions import Uniform
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UniformDiscrete(Distribution):
    """
    A uniform discrete distribution over a finite set of initial states.
    """

    def __init__(self, support: Iterable[str]):
        super().__init__()
        self.values = list(support)
        self.categorical = Categorical(
            probs=torch.ones(len(self.values)) / len(self.values)
        )
        self.stochastic = True

    def enumerate_support(self, current_time_step: int = 0, expand=False, **kwargs):
        """
        Enumerate the support of the distribution.
        """
        return self.values

    def sample(self, sample_shape=torch.Size([]), **kwargs) -> str:
        """
        Sample from the distribution.
        """
        return self.values[self.categorical.sample(sample_shape).tolist()]

    def sample_n(self, k: int, **kwargs) -> Iterable[str]:
        """
        Sample n times from the distribution.
        """
        indices = self.categorical.sample((k,)).tolist()
        return [self.values[i] for i in indices]


class InstructTemplateDistribution(Distribution):
    """
    A distribution that can be sampled, which is based on a template with parameters.
    """

    def __init__(
        self,
        template: dict,
        parameters: Iterable[Distribution],
        system_prompt: Iterable = None,
        validate_args=False,
    ):
        super().__init__(validate_args=False)

        # e.g. {"role": "user"} or a list of these if doing few-shot prompting.
        # The shots should be deterministic and the last list entry should be templatic.
        self.template = template

        self.arg_distributions = (
            parameters  # List of distributions for each parameter in each {}
        )
        self.system_prompt = system_prompt  # Optional system prompt for the LLM
        self.stochastic = True

    def fill_template(self, param_list):
        template = self.template.copy()
        template["content"] = self.template["content"].format(*param_list)

        if self.system_prompt is not None:
            return [
                *self.system_prompt,
                template,
            ]  # Making it compatible with the LLM interface
        else:
            return [template]

    def enumerate_support(self, current_time_step: int = 0, expand=False, **kwargs):
        """
        Enumerate the support of the distribution.
        """
        logger.info("Generated text from template: {}".format(self.template))

        supports = []
        for distn in self.arg_distributions:
            # Uniform (continuous) distributions' support is continuous so can't be enumerated
            if type(distn) == Uniform:
                return

            distn_support = distn.enumerate_support()

            if distn_support is None:
                return

            elif isinstance(distn_support, list):
                supports.append(distn_support)

        # All supports are enumerable
        all_parameter_combinations = [list(p) for p in itertools.product(*supports)]

        return [self.fill_template(params) for params in all_parameter_combinations]

    def sample(self, sample_shape=torch.Size([]), **kwargs) -> dict:
        """
        Sample from the distribution.
        """
        # For simplicity, we just return the template with the first parameter
        # In practice, you would sample from the parameters
        if len(self.arg_distributions) == 0:
            return self.template

        # Fill in the template with sampled parameters
        sampled_params = [dist.sample() for dist in self.arg_distributions]

        if type(sampled_params[0]) == torch.Tensor:
            sampled_params = [param.tolist() for param in sampled_params]

        filled_template = self.fill_template(sampled_params)

        return filled_template

    def sample_n(self, k: int, **kwargs) -> Iterable[dict]:
        """
        Sample k times from the distribution.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")

        return [self.sample() for _ in range(k)]


class StringTemplateDistribution(Distribution):
    """
    A uniform distribution supported over a string template + parameters.
    Ex. "Generate an image with saturation {Unif[0, 1]}".
    """

    def __init__(
        self,
        template: str,
        parameters: Iterable[Distribution],
        validate_args=False,
        formatters: dict[str, Any] = None,
    ):
        """
        Initialize the uniform continuous distribution.
            :param template: The string template with placeholders for parameters. E.g. "Generate an image with saturation {}".
            :param parameters: List of distributions for each parameter. The length is the same as the number of {}
            :param arg_constraints: Constraints on the arguments (optional).
        """
        super().__init__(validate_args=False)
        self.template = template  # The string template with placeholders for parameters
        self.arg_distributions = parameters  # List of distributions for each parameter
        self.stochastic = True

        self.formatters = formatters if formatters is not None else {}
        if "object" in self.formatters:
            self.inflect_engine = inflect.engine()

    def _should_pluralize(self, x):
        """
        Returns True if:
          - x is a decimal strictly between 0 and 1, or
          - x is exactly 1.
        Returns False if:
          - x is an integer greater than 1.
        """
        if not isinstance(x, (int, float)):
            return False
        if x == 1:
            return False
        if 0 < x < 1:
            return False
        if isinstance(x, int) and x > 1:
            return True
        return False

    def fill_template(self, param_list):
        formatters = copy.deepcopy(self.formatters)
        if "object" in self.formatters and self._should_pluralize(param_list[0]):
            formatters["object"] = self.inflect_engine.plural(formatters["object"])
        return self.template.format(*param_list, **formatters)

    def enumerate_support(self, current_time_step: int = 0, **kwargs):
        logger.info("Generated text from template: {}".format(self.template))
        supports = []
        for distn in self.arg_distributions:
            # Uniform (continuous, from torch.distributions) doesn't have enumerate support
            # because its support is continuous.
            if type(distn) == Uniform:
                return

            distn_support = distn.enumerate_support()

            if distn_support is None:
                return

            elif isinstance(distn_support, Iterable):
                supports.append(distn_support)

        # All supports are enumerable
        all_parameter_combinations = [list(p) for p in itertools.product(*supports)]

        return [self.fill_template(params) for params in all_parameter_combinations]

    def sample(self, sample_shape=torch.Size([]), **kwargs) -> str:
        """
        Sample from the distribution.
        """
        # For simplicity, we just return the template with the first parameter
        # In practice, you would sample from the parameters
        if len(self.arg_distributions) == 0:
            return self.template

        sampled_params = [dist.sample() for dist in self.arg_distributions]
        return self.fill_template(sampled_params)

    def sample_n(self, k: int, **kwargs) -> Iterable[str]:
        """
        Sample k times from the distribution.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")

        return [self.sample() for _ in range(k)]


class LLM(Distribution):
    """
    A distribution that can be sampled, which is based on an LLM given input prompt.
    """

    arg_constraints = {}

    def __init__(
        self,
        model_name: str,
        input_prompt: str,
        sampling_config: dict = {},
    ):
        super().__init__()
        self.model_name = model_name
        self.input_prompt = input_prompt
        self.stochastic = True

        # Load the model. We require that the model is a chat-tuned one!
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            device=device,
            max_length=512,
            **sampling_config,
        )

    def enumerate_support(self, current_time_step: int = 0, expand=False):
        """
        Enumerate the support of the distribution.
        """
        logger.info(
            "Generated text from {} with prompt {}".format(
                self.model_name, self.input_prompt
            )
        )
        return

    def sample(self, sample_shape=torch.Size([]), **kwargs) -> str:
        return self.generator(self.input_prompt, num_return_sequences=1)[0][
            "generated_text"
        ].lstrip()

    def sample_n(self, k: int, **kwargs) -> Iterable[str]:
        """
        Sample k times from the distribution.
        """
        # Just get the text from the generator
        if k <= 0:
            raise ValueError("k must be a positive integer")
        return [
            response["generated_text"].lstrip()
            for response in self.generator(self.input_prompt, num_return_sequences=k)
        ]


# Everything below is a conditional distribution!
class ConditionalDistribution(Distribution):
    """
    This is a conditional distribution that you can sample conditioned on some variables.
    Main usage: an input that is conditioned on previous states and outputs as a form of feedback.

    u ~ Distribution(previous states and outputs)

    It should be flexible enough to implement a deterministic function u(previous x, previous y), as well as
    a distribution based on an LLM.
    """

    arg_constraints = {}

    def __init__(self, feedback_function: callable):
        super().__init__()
        self.feedback_function = feedback_function
        self.stochastic = False

    def sample(self, sample_shape=torch.Size([]), **condition_variables) -> str:
        batched_condition_variables = [
            dict(zip(condition_variables.keys(), values))
            for values in zip(*condition_variables.values())
        ]
        return [
            self.feedback_function(**cond_i) for cond_i in batched_condition_variables
        ]

    def sample_n(self, k: int, **batched_condition_variables) -> Iterable[str]:
        samples = []
        for i in range(k):
            # pick the i-th element from each condition variable
            cond_i = {
                name: values[i] for name, values in batched_condition_variables.items()
            }
            samples.append(self.sample(**cond_i))
        return samples

    def enumerate_support(
        self, current_time_step: int = 0, **condition_variables
    ) -> Iterable:
        """
        This distribution is the pushforward of a previous one
        (condition_variables in the function statement).
        So the support is just taking all possible condition_variables and
        evaluating them with feedback_function.

        Require:
        condition_variables {keyword: [values]} enumerated as the grid of all possible
        combinations.

        Returns:
        The grid evaluted.
        """
        name = condition_variables.keys()[0]
        grid_size = len(condition_variables[name])
        return self.sample_n(grid_size, **condition_variables)


class LLMConditionalDistribution(ConditionalDistribution):
    def __init__(
        self, feedback_function: callable, model_name: str, sampling_config: dict = {}
    ):
        super().__init__(feedback_function)
        self.model_name = model_name
        self.stochastic = True

        # Load the model. We require that the model is a chat-tuned one!
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            model=model_name, device=device, max_length=512, **sampling_config
        )

    def sample(self, sample_shape=torch.Size([]), **condition_variables) -> str:
        # Build the prompt based on whatever variables there are (e.g. prev state, prev output)
        input_prompt = self.feedback_function(**condition_variables)
        return self.generator(input_prompt, num_return_sequences=1)[0][
            "generated_text"
        ].lstrip()

    def sample_n(self, k: int, **batched_condition_variables) -> str:
        # Build the prompts
        input_prompts = []
        for i in range(k):
            # pick the i-th element from each condition variable
            cond_i = {
                name: values[i] for name, values in batched_condition_variables.items()
            }
            input_prompts.append(self.feedback_function(**cond_i))

        generations = self.generator(input_prompts, num_return_sequences=1)
        inputs = [generation["generated_text"].lstrip() for generation in generations]

        return inputs  # k strings that are the next input

    def enumerate_support(self, current_time_step: int = 0, **kwargs) -> Iterable:
        """
        Enumerates the support. Since we're sampling from an LLM, the support is huge.
        So default to infinity (returning None)
        """
        return None


class DialogueInputDistribution(Distribution):
    """
    This contains an ensemble of time-dependent distributions.
    """

    def __init__(self, distributions: Iterable[Distribution], indexer: callable):
        super().__init__(validate_args=False)

        # A function t -> Distributions that chooses the right distribution to
        # sample from depending on the timestep.
        self.indexer = indexer  # function taking the timestep to the distribution
        self.distributions = distributions  # list of distributions u1, u2, ...

    def sample(self, timestep: int = 0, **condition_variables) -> str:
        # Pick u_t, the correct distribution for the current timestep
        distribution_t = self.indexer(timestep)
        assert distribution_t in self.distributions
        return distribution_t.sample(**condition_variables)

    def sample_n(self, sample_size: int, timestep: int, **condition_variables) -> str:
        # Pick u_t, the correct distribution for the current timestep
        distribution_t = self.indexer(timestep)
        return distribution_t.sample_n(sample_size, **condition_variables)

    def enumerate_support(
        self, current_time_step: int = 0, **condition_variables
    ) -> Iterable:
        # The support size is always the same.
        current_input_distribution = self.indexer(current_time_step)
        return current_input_distribution.enumerate_support(**condition_variables)

    def is_stochastic(self, current_time_step: int = 0) -> bool:
        return self.indexer(current_time_step).stochastic


if __name__ == "__main__":
    # Example usage
    initial_states = UniformDiscrete(["Hello", "Hi", "Greetings"])
    print("Sampled initial state:", initial_states.sample())
    print("Sampled 5 initial states:", list(initial_states.sample_n(5)))

    sampling_config = {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.7,
        "return_full_text": False,
    }
    distn = LLM(
        "mistralai/Mistral-7B-Instruct-v0.1",
        "Imagine you are a user about to ask a question to a chatbot. Say your opening greeting and only the greeting.",
        sampling_config=sampling_config,
    )

    # print(distn.sample())
    print(distn.sample_n(500))
