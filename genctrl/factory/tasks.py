# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""
Task-based architecture for genctrl.
Each task encapsulates its own input space, initial states, output space, and feedback functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

from omegaconf import OmegaConf
from torch.distributions.distribution import Distribution

from genctrl.factory.initial_states_library import INITIAL_STATES_FACTORY
from genctrl.factory.input_space_library import (
    FEEDBACK_FACTORY,
    INPUT_SPACE_FACTORY,
    INPUT_VALUE_FACTORY,
)
from genctrl.factory.outputs_library import (
    OUTPUT_SPACE_FACTORY,
    OUTPUTS_FACTORY,
)


class Task(ABC):
    """
    Base class for all tasks.

    Each task defines:
    - Input space: how to generate inputs/prompts
    - Initial states: starting states for the model
    - Output space: what the valid outputs are
    - Output map: how to evaluate model outputs
    - Feedback: (optional) how to provide feedback in dialogue settings
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the task with optional configuration.

        Args:
            config: Dictionary containing task-specific configuration parameters.
                   Can include:
                   - initial_states.initial_states_str: How to get initial states
                   - initial_states.instruction_tuned: Whether the task uses instruction tuning (default: True)
                   - initial_states.num_shots: Number of few-shot examples (0, 1, or 5) (default: 0)
                   - initial_states.use_system_prompt: Whether to use system prompt (default: False)
                   - input_space.start, input_space.end, input_space.obj, etc.: Task-specific parameters
        """
        self.config = config or {}
        self.initial_states_str = OmegaConf.select(self.config, "initial_states.initial_states_str", default="empty")

    @abstractmethod
    def get_input_space(self, **kwargs) -> tuple:
        """
        Get the input space specification.

        Returns:
            Tuple containing template and argument distributions.
            For instruction-tuned tasks, also returns system prompt.
        """
        pass

    def get_initial_states(self, **kwargs) -> list[str]:
        """
        Get the initial states for this task.

        Returns:
            List of initial state strings
        """
        return INITIAL_STATES_FACTORY.get(self.initial_states_str, [""])

    @abstractmethod
    def get_output_map(self, **kwargs) -> Callable:
        """
        Get the output map function for evaluating model outputs.

        Returns:
            Callable that maps model outputs to evaluated values
        """
        pass

    @abstractmethod
    def get_output_space(self, **kwargs) -> Any:
        """
        Get the output space (valid output values).

        Returns:
            Set or list of valid output values
        """
        pass

    def get_feedback_function(self, **kwargs) -> Optional[Callable]:
        """
        Get the feedback function for dialogue settings (optional).

        Returns:
            Callable that generates feedback given initial inputs and last outputs,
            or None if this task doesn't support feedback
        """
        return None

    def get_value_extractor(self) -> Optional[Callable]:
        """
        Get the function to extract target values from input strings.

        Returns:
            Callable that extracts the target value from an input string,
            or None if not applicable
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the base task name (e.g., 'num_chars', 'even_odd')."""
        pass

    @property
    def supports_dialogue(self) -> bool:
        """Return whether this task supports dialogue/feedback."""
        return self.get_feedback_function() is not None

    @property
    def is_dialogue(self) -> bool:
        """Return whether this run uses dialogue (time_steps > 1)."""
        time_steps = OmegaConf.select(self.config, 'time_steps', default=1)
        return time_steps > 1

    @property
    def instruction_tuned(self) -> bool:
        """Return whether this task uses instruction tuning."""
        return OmegaConf.select(self.config, 'initial_states.instruction_tuned', default=True)

    @property
    def num_shots(self) -> int:
        """Return the number of few-shot examples (0, 1, or 5)."""
        return OmegaConf.select(self.config, 'initial_states.num_shots', default=0)

    @property
    def use_system_prompt(self) -> bool:
        """Return whether this task uses a system prompt."""
        return OmegaConf.select(self.config, 'initial_states.use_system_prompt', default=False)

    def _get_input_space_key(self) -> str:
        """
        Build the key for INPUT_SPACE_FACTORY lookup.

        Returns:
            String key based on task name and configuration (instruction_tuned, num_shots, use_system_prompt)
        """
        key = self.name
        if self.instruction_tuned:
            key += "_it"
            if self.num_shots > 0:
                key += f"_{self.num_shots}shot"
            if self.use_system_prompt:
                key += "_system"
        return key


class NumCharsTask(Task):
    """Task for generating strings with a specific number of characters."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.start = OmegaConf.select(self.config, "input_space.start", default=1)
        self.end = OmegaConf.select(self.config, "input_space.end", default=10)

    @property
    def name(self) -> str:
        return "num_chars"

    def get_input_space(self, **kwargs):
        key = self._get_input_space_key()
        if key in INPUT_SPACE_FACTORY:
            return INPUT_SPACE_FACTORY[key](self.start, self.end)
        raise ValueError(f"Input space configuration not found: {key}")

    def get_output_map(self, **kwargs):
        return OUTPUTS_FACTORY["num_chars"]

    def get_output_space(self, **kwargs):
        return OUTPUT_SPACE_FACTORY["num_chars"](bounds=(self.start, self.end))

    def get_feedback_function(self, **kwargs) -> Optional[Callable]:
        if self.is_dialogue and "num_chars" in FEEDBACK_FACTORY:
            return FEEDBACK_FACTORY["num_chars"]
        return None

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("num_chars")


class EvenOddTask(Task):
    """Task for generating even or odd integers."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "even_odd"

    def get_input_space(self, **kwargs):
        key = self._get_input_space_key()
        if key in INPUT_SPACE_FACTORY:
            return INPUT_SPACE_FACTORY[key]()
        raise ValueError(f"Input space configuration not found: {key}")

    def get_output_map(self, **kwargs):
        return OUTPUTS_FACTORY["even_odd"]

    def get_output_space(self, **kwargs):
        return OUTPUT_SPACE_FACTORY["even_odd"]()

    def get_feedback_function(self, **kwargs) -> Optional[Callable]:
        if self.is_dialogue and "even_odd" in FEEDBACK_FACTORY:
            return FEEDBACK_FACTORY["even_odd"]
        return None

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("even_odd")


class FormalityTask(Task):
    """Task for generating text with specific formality level."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.start = OmegaConf.select(self.config, "input_space.start", default=0.0)
        self.end = OmegaConf.select(self.config, "input_space.end", default=1.0)

    @property
    def name(self) -> str:
        return "formality"

    def get_input_space(self, **kwargs):
        key = self._get_input_space_key()
        if key in INPUT_SPACE_FACTORY:
            return INPUT_SPACE_FACTORY[key](self.start, self.end)
        raise ValueError(f"Input space configuration not found: {key}")

    def get_output_map(self, **kwargs):
        # Formality requires a neural scorer
        raise NotImplementedError("Formality task requires a neural scorer model")

    def get_output_space(self, **kwargs):
        return [[0, 1]]  # Normalized formality score

    def get_feedback_function(self, **kwargs) -> Optional[Callable]:
        if self.is_dialogue and "formality" in FEEDBACK_FACTORY:
            return FEEDBACK_FACTORY["formality"]
        return None

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("formality")


class AverageWordLengthTask(Task):
    """Task for generating sentences with specific average word length."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.start = OmegaConf.select(self.config, "input_space.start", default=2)
        self.end = OmegaConf.select(self.config, "input_space.end", default=10)

    @property
    def name(self) -> str:
        return "average_word_length"

    def get_input_space(self, **kwargs):
        key = self._get_input_space_key()
        if key in INPUT_SPACE_FACTORY:
            return INPUT_SPACE_FACTORY[key](self.start, self.end)
        raise ValueError(f"Input space configuration not found: {key}")

    def get_output_map(self, **kwargs):
        return OUTPUTS_FACTORY["average_word_length"]

    def get_output_space(self, **kwargs):
        return OUTPUT_SPACE_FACTORY["average_word_length"]()

    def get_feedback_function(self, **kwargs) -> Optional[Callable]:
        if self.is_dialogue and "average_word_length" in FEEDBACK_FACTORY:
            return FEEDBACK_FACTORY["average_word_length"]
        return None

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("average_word_length")


class WhiteBackgroundObjectsTask(Task):
    """Task for generating images with specific number of objects on white background."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.start = OmegaConf.select(self.config, "input_space.start", default=1)
        self.end = OmegaConf.select(self.config, "input_space.end", default=15)
        self.obj = OmegaConf.select(self.config, "input_space.obj", default="cat")

    @property
    def name(self) -> str:
        return "white_bg_objects"

    @property
    def instruction_tuned(self) -> bool:
        """Text-to-image tasks are not instruction tuned."""
        return False

    def get_input_space(self, **kwargs):
        return INPUT_SPACE_FACTORY["white_bg_objects"](self.start, self.end)

    def get_output_map(self, **kwargs):
        # This task requires an object detection model.
        # The output map should be configured via output_map_str in the config,
        # which will be handled by utils.get_output_map()
        raise NotImplementedError(
            "WhiteBackgroundObjectsTask requires an object detection model. "
            "Please specify 'output_map_str' in your config (e.g., 'google/owlv2-base-patch16-ensemble')"
        )

    def get_output_space(self, **kwargs):
        return OUTPUT_SPACE_FACTORY["objects"](bounds=(self.start, self.end))

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("white_bg_objects")


class WhiteBackgroundPositionTask(Task):
    """Task for generating images with object at specific position."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.obj = OmegaConf.select(self.config, "input_space.obj", default="cat")

    @property
    def name(self) -> str:
        return "white_bg_position_objects"

    @property
    def instruction_tuned(self) -> bool:
        """Text-to-image tasks are not instruction tuned."""
        return False

    def get_input_space(self, **kwargs):
        return INPUT_SPACE_FACTORY["white_bg_position_objects"]()

    def get_output_map(self, **kwargs):
        return OUTPUTS_FACTORY["object_position"]

    def get_output_space(self, **kwargs):
        return OUTPUT_SPACE_FACTORY["object_position"]()

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("white_bg_position_objects")


class SaturationTask(Task):
    """Task for generating images with specific saturation level."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.start = OmegaConf.select(self.config, "input_space.start", default=0.0)
        self.end = OmegaConf.select(self.config, "input_space.end", default=1.0)

    @property
    def name(self) -> str:
        return "saturation"

    @property
    def instruction_tuned(self) -> bool:
        """Text-to-image tasks are not instruction tuned."""
        return False

    def get_input_space(self, **kwargs):
        return INPUT_SPACE_FACTORY["saturation"](self.start, self.end)

    def get_output_map(self, **kwargs):
        return OUTPUTS_FACTORY["saturation"]

    def get_output_space(self, **kwargs):
        return OUTPUT_SPACE_FACTORY["saturation"]()

    def get_value_extractor(self):
        return INPUT_VALUE_FACTORY.get("saturation")


# Task registry
TASK_REGISTRY = {
    "num_chars": NumCharsTask,
    "even_odd": EvenOddTask,
    "formality": FormalityTask,
    "average_word_length": AverageWordLengthTask,
    "white_bg_objects": WhiteBackgroundObjectsTask,
    "white_bg_position_objects": WhiteBackgroundPositionTask,
    "saturation": SaturationTask,
}


def create_task(task_name: str, config: Optional[dict] = None) -> Task:
    """
    Factory function to create a task instance from a name and config.

    Args:
        task_name: Base name of the task (e.g., "num_chars", "even_odd", "white_bg_objects")
        config: Optional configuration dictionary. Configuration should specify:
                - initial_states.instruction_tuned: Whether to use instruction tuning
                - initial_states.num_shots: Number of few-shot examples (0, 1, or 5)
                - initial_states.use_system_prompt: Whether to use system prompt
                - input_space parameters (start, end, obj, etc.)

    Returns:
        Task instance

    Example:
        >>> task = create_task("num_chars", {"input_space": {"start": 1, "end": 20}, "initial_states": {"instruction_tuned": True}})
        >>> task.name  # Returns "num_chars"
        >>> task.instruction_tuned  # Returns True
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}"
        )

    if config is None:
        config = {}

    task_class = TASK_REGISTRY[task_name]
    return task_class(config)


def list_tasks() -> list[str]:
    """Return list of available task names."""
    return list(TASK_REGISTRY.keys())
