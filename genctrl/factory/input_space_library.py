# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import re

import inflect
import numpy as np
import tree
from torch.distributions import Uniform
from torch.distributions.distribution import Distribution

from genctrl.utils.distributions import (
    ConditionalDistribution,
    LLMConditionalDistribution,
    UniformDiscrete,
)


# INPUT SPACE FOR STRINGS ============================================================
def num_chars(start: int = 1, end: int = 10) -> tuple[str, list[Distribution]]:
    template = "Write a string of {} characters. Strictly write the string without quotes and do not write anything else."
    arg_distributions = [UniformDiscrete(list(range(start, end + 1)))]
    return template, arg_distributions


def num_chars_instruction_tuned(
    start: int = 1, end: int = 11
) -> tuple[str, list[Distribution]]:
    content_str = "Write a string of {} characters. Strictly write the string without quotes and do not write anything else."
    arg_distributions = [UniformDiscrete(list(range(start, end + 1)))]

    template = {"role": "user", "content": content_str}
    return template, arg_distributions, None


def num_chars_instruction_tuned_1shot(
    start: int = 1, end: int = 11
) -> tuple[str, list[Distribution]]:
    content_str = """Instruction: Write a string of 2 characters. 
    Result:
    Hi

    Instruction: Write a string of {} characters. Strictly write the string without quotes and do not write anything else.
    Result:
    """
    arg_distributions = [UniformDiscrete(list(range(start, end + 1)))]

    template = {"role": "user", "content": content_str}
    return template, arg_distributions, None


def num_chars_instruction_tuned_5shot(
    start: int = 1, end: int = 10
) -> tuple[str, list[Distribution]]:
    content_str = """You are a chatbot that follows instructions. I will give you some examples:


        Instruction: Write a string of 2 characters. 
        Result:
        Hi

        Instruction: Write a string of 5 characters.
        Result:
        Hello

        Instruction: Write a string of 7 characters. 
        Result:
        abcdefg

        Instruction: Write a string of 4 characters.
        Result:
        Cars

        Instruction: Write a string of 10 characters.
        classifier 

    Knowing this, please follow the next instruction.

    Instruction: Write a string of {} characters. Strictly write the string without quotes and do not write anything else."""

    arg_distributions = [UniformDiscrete(list(range(start, end + 1)))]

    template = {"role": "user", "content": content_str}

    return template, arg_distributions, None


def num_chars_instruction_tuned_5shot_system(start: int = 1, end: int = 10):
    content_str = "Write a string of {} characters. Strictly write the string without quotes and do not write anything else."

    # Few-shot prompting
    system_prompt = [
        {
            "role": "user",
            "content": "Write a string of 2 characters. Strictly write the string without quotes and do not write anything else.",
        },
        {"role": "assistant", "content": "Hi"},
        {
            "role": "user",
            "content": "Write a string of 5 characters. Strictly write the string without quotes and do not write anything else.",
        },
        {"role": "assistant", "content": "Hello"},
        {
            "role": "user",
            "content": "Write a string of 7 characters. Strictly write the string without quotes and do not write anything else.",
        },
        {"role": "assistant", "content": "abcdefg"},
        {
            "role": "user",
            "content": "Write a string of 4 characters. Strictly write the string without quotes and do not write anything else.",
        },
        {"role": "assistant", "content": "Cars"},
        {
            "role": "user",
            "content": "Write a string of 10 characters. Strictly write the string without quotes and do not write anything else.",
        },
        {"role": "assistant", "content": "classifier"},
    ]

    arg_distributions = [UniformDiscrete(list(range(start, end + 1)))]

    template = {"role": "user", "content": content_str}
    return template, arg_distributions, system_prompt


def num_chars_instruction_tuned_system(
    start: int = 1, end: int = 11
) -> tuple[str, list[Distribution]]:
    sys_prompt = [
        {
            "role": "system",
            "content": f"You are a chatbot that strictly follows user requests.",
        }
    ]
    template, arg_distributions = num_chars_instruction_tuned(start, end)
    return template, arg_distributions, sys_prompt


def even_odd_instruction_tuned() -> tuple[str, list[Distribution]]:
    content_str = "Generate a strictly positive {} integer."
    arg_distributions = [UniformDiscrete(["odd", "even"])]

    template = {"role": "user", "content": content_str}
    return template, arg_distributions, None


def even_odd_instruction_tuned_1shot_system() -> tuple[str, list[Distribution]]:
    content_str = "Generate a strictly positive {} integer. Only generate the integer and nothing else."

    arg_distributions = [UniformDiscrete(["odd", "even"])]

    template = {"role": "user", "content": content_str}

    system_prompt = [
        {
            "role": "user",
            "content": "Generate a strictly positive odd integer. Only generate the integer and nothing else.",
        },
        {"role": "assistant", "content": "31"},
    ]
    return template, arg_distributions, system_prompt


def even_odd_instruction_tuned_1shot() -> tuple[str, list[Distribution]]:
    content_str = """You are a chatbot that follows instructions. I will give you an example:
        Instruction: Generate a strictly positive odd integer. Only generate the integer and nothing else.
        Result:
        31

        Knowing this, please follow the next instruction.

        Instruction: Generate a strictly positive {} integer. Only generate the integer and nothing else.
    """
    arg_distributions = [UniformDiscrete(["odd", "even"])]

    template = {"role": "user", "content": content_str}

    return template, arg_distributions, None


def even_odd_instruction_tuned_5shot() -> tuple[str, list[Distribution]]:
    content_str = """You are a chatbot that follows instructions. I will give you some examples:
        Instruction: Generate a strictly positive odd integer. Only generate the integer and nothing else.
        Result:
        31

        Instruction: Generate a strictly positive even integer. Only generate the integer and nothing else.
        Result:
        94726

        Instruction: Generate a strictly positive odd integer. Only generate the integer and nothing else.
        Result:
        7

        Instruction: Generate a strictly positive even integer. Only generate the integer and nothing else.
        Result:
        200

        Instruction: Generate a strictly positive odd integer. Only generate the integer and nothing else.
        Result:
        2399

        Knowing this, please follow the next instruction.

        Instruction: Generate a strictly positive {} integer. Only generate the integer and nothing else.
    """
    arg_distributions = [UniformDiscrete(["odd", "even"])]

    template = {"role": "user", "content": content_str}

    return template, arg_distributions, None


def even_odd_instruction_tuned_5shot_system() -> tuple[str, list[Distribution]]:
    content_str = """Generate a strictly positive {} integer. Only generate the integer and nothing else.
    """
    arg_distributions = [UniformDiscrete(["odd", "even"])]

    template = {"role": "user", "content": content_str}

    # Shots in few-shot prompting
    system_prompt = [
        {
            "role": "user",
            "content": "Generate a strictly positive odd integer. Only generate the integer and nothing else.",
        },
        {"role": "assistant", "content": "31"},
        {
            "role": "user",
            "content": "Generate a strictly positive even integer. Only generate the integer and nothing else.",
        },
        {"role": "assistant", "content": "94726"},
        {
            "role": "user",
            "content": "Generate a strictly positive odd integer. Only generate the integer and nothing else.",
        },
        {"role": "assistant", "content": "7"},
        {
            "role": "user",
            "content": "Generate a strictly positive even integer. Only generate the integer and nothing else.",
        },
        {"role": "assistant", "content": "200"},
        {
            "role": "user",
            "content": "Generate a strictly positive odd integer. Only generate the integer and nothing else.",
        },
        {"role": "assistant", "content": "2399"},
    ]

    return template, arg_distributions, system_prompt


def empty_instruction_tuned() -> tuple[str, list[Distribution]]:
    template = {"role": "user", "content": ""}

    return template, [], None


def formality_it(start: float = 0, end: float = 1) -> tuple[dict, list[Distribution]]:
    content_str = (
        "Generate a story that is {:.2f}% formal. Write the story and only the story."
    )
    arg_distribution = [Uniform(start * 100, end * 100)]

    template = {"role": "user", "content": content_str}
    return template, arg_distribution, None


def formality_it_1shot(
    start: float = 0, end: float = 1
) -> tuple[dict, list[Distribution]]:
    content_str = """You are a chatbot that follows instructions. I will give you an example:
    Instruction: Generate a story that is 100% formal.
    Result:
    Would you please kindly knock when needed, sir?    

    Knowing this, please follow the next instruction.

    Generate a story that is {:.2f}% formal. Write the story and only the story."""
    arg_distribution = [Uniform(start * 100, end * 100)]

    template = {"role": "user", "content": content_str}
    return template, arg_distribution, None


def formality_it_5shot(
    start: float = 0, end: float = 1
) -> tuple[dict, list[Distribution]]:
    content_str = """You are a chatbot that follows instructions. I will give you an example:
    Instruction: Generate a story that is 100% formal.
    Result:
    Kind sir, please make your way to the terrace.  

    Instruction: Generate a story that is 75% formal.
    Result:
    She bought the book, then she went to the beach.

    Instruction: Generate a story that is 50% formal.
    Result:
    Hey! Just went to the store.

    Instruction: Generate a story that is 25% formal.
    Result:
    Wow I'm so tired.

    Instruction: Generate a story that is 0% formal.
    Result:
    hey man how's it goin

    Knowing this, please follow the next instruction.

    Generate a story that is {:.2f}% formal. Write the story and only the story."""
    arg_distribution = [Uniform(start * 100, end * 100)]

    template = {"role": "user", "content": content_str}
    return template, arg_distribution, None


def average_word_length_it(
    start: int = 2, end: int = 10
) -> tuple[dict, list[Distribution]]:
    content_str = """Generate a sentence where the average word length is exactly {:0.2f} letters. Strictly write the sentence without quotes and do not write anything else."""

    arg_distribution = [Uniform(start, end)]

    template = {"role": "user", "content": content_str}
    return template, arg_distribution, None


def average_word_length_it_1shot(
    start: int = 2, end: int = 10
) -> tuple[dict, list[Distribution]]:
    content_str = """You are a chatbot that follows instructions. I will give you an example:

    Instruction: Generate a sentence where the average word length is exactly 4 letters. 
    Result:
    The pig jumped over them.

    Knowing this, please follow the next instruction.

    Generate a sentence where the average word length is exactly {:.2f} letters. Strictly write the sentence without quotes and do not write anything else."""

    arg_distribution = [Uniform(start, end)]

    template = {"role": "user", "content": content_str}
    return template, arg_distribution, None


def average_word_length_it_5shot(
    start: int = 2, end: int = 10
) -> tuple[dict, list[Distribution]]:
    rng = np.random.default_rng(seed=42)
    shots = rng.uniform(start, end, 5)
    content_str = f"""You are a chatbot that follows instructions. I will give you an example:

    Instruction: Generate a sentence where the average word length is exactly {shots[0]:0.1f} letters. 
    Result:
    The pig jumped above her.

    Instruction: Generate a sentence where the average word length is exactly {shots[1]:0.1f} letters.
    Result:
    She bought the book, then she went to the beach.

    Instruction: Generate a sentence where the average word length is exactly {shots[2]:0.1f} letters.
    Result:
    Hey! Just went to the store.

    Instruction: Generate a sentence where the average word length is exactly {shots[3]:0.1f} letters.
    Result:
    Wow I'm so tired.

    Instruction: Generate a sentence where the average word length is exactly {shots[4]:0.1f} letters.
    Result:
    Stop redistribution!

    Knowing this, please follow the next instruction.

    Generate a sentence where the average word length is exactly {{:.2f}} letters. Strictly write the sentence without quotes and do not write anything else."""

    arg_distribution = [Uniform(start, end)]

    template = {"role": "user", "content": content_str}
    return template, arg_distribution, None


# FEEDBACK FUNCTION U(INITIAL_INPUTS, LAST_OUTPUTS) -> next input, stop or continue
def template_dialogue(initial_inputs: str, last_outputs) -> tuple[str, bool]:
    # Step 1: extract the desired value from the first inputs

    # Step 2: Evaluate whether the goal was reached

    # Step 3: stop condition

    # Step 4: write the feedback

    pass


def even_odd_dialogue(initial_inputs: str, last_outputs) -> tuple[str, bool]:
    # The last_outputs should have an output space in [True, False], check the outputs library.
    # True maps onto even.

    # Step 1: extract the desired value from the first inputs
    desired_value = INPUT_VALUE_FACTORY["even_odd"](initial_inputs[0]["content"])

    # Step 2: Evaluate whether the goal was reached
    if last_outputs == True:
        last_str = "even number"
    elif last_outputs == False:
        last_str = "odd number"
    elif last_outputs == "error":
        last_str = "noninteger string"

    # Step 3: stop condition
    stop = desired_value == last_outputs

    # Step 4: feedback input
    feedback = [
        {
            "role": "user",
            "content": f"Your answer was incorrect. I asked for a {desired_value} number, and you gave an {last_str}. Please try again. Strictly write the integer without quotes and do not write anything else.",
        }
    ]

    return feedback, stop


def formality_dialogue(initial_inputs: str, last_outputs) -> str:
    # Step 1: extract the desired value from the first inputs
    desired_value = INPUT_VALUE_FACTORY["formality"](initial_inputs[0]["content"])

    # Step 2: Evaluate whether the goal was reached
    formal_str = "formal" if desired_value < last_outputs * 100 else "informal"

    # Step 3: Stop condition
    stop = desired_value == last_outputs

    # Step 4: write the feedback
    feedback = [
        {
            "role": "user",
            "content": f"Your answer was too {formal_str}. I asked for a formality level of {desired_value:.2f}, and you gave a sentence with formality {100 * last_outputs:.2f}. Please try again. Write the story and only the story.",
        }
    ]

    return feedback, stop


def num_chars_dialogue(initial_inputs, last_outputs) -> str:
    # Step 1: extract the desired value from the first inputs
    desired_value = INPUT_VALUE_FACTORY["num_chars"](initial_inputs[0]["content"])

    # Step 2: Evaluate whether the goal was reached
    custom_str = "long" if desired_value < last_outputs else "short"

    # Step 3: Stop condition
    stop = desired_value == last_outputs

    # Step 4: write the feedback
    feedback = [
        {
            "role": "user",
            "content": f"Your answer was too {custom_str}. I asked for a string consisting of {desired_value} characters, and you produced one with {last_outputs} characters. Please try again. Strictly write the string without quotes and do not write anything else.",
        }
    ]
    return feedback, stop


def average_word_length_dialogue(initial_inputs, last_outputs) -> str:
    # Step 1: extract the desired value from the first inputs
    desired_value = INPUT_VALUE_FACTORY["average_word_length"](
        initial_inputs[0]["content"]
    )
    custom_str = "long" if last_outputs > desired_value else "short"

    # Step 3: Stop condition
    stop = desired_value == last_outputs

    feedback = [
        {
            "role": "user",
            "content": f"Your answer was too {custom_str}. I asked for a string whose average word length is {desired_value:.2f} letters, and you produced one with average word length {last_outputs:.2f}. Please try again. Strictly write the sentence without quotes and do not write anything else.",
        }
    ]

    return feedback, stop


# INPUT SPACE FOR IMAGES ============================================================
def saturation(start: float = 0, end: float = 1) -> tuple[str, list[Distribution]]:
    template = "Generate an image with {:.2f}% saturation."
    arg_distribution = [Uniform(start * 100, end * 100)]
    return template, arg_distribution


def white_bg_objects(
    start: int = 1, end: int = 15, **unused_kwargs
) -> tuple[str, list[Distribution]]:
    # template = "White background. {} " + obj + "s."
    template = "White background. {} {object}."
    # template = "{} " + obj + "s on a white background."
    arg_distribution = [UniformDiscrete(list(range(start, end + 1)))]
    return template, arg_distribution


def white_bg_position_objects(**unused_kwargs) -> tuple[str, list[Distribution]]:
    template = "White background. A {object} at the {} of the image."
    choices = ["top left", "top right", "bottom left", "bottom right", "center"]
    arg_distribution = [UniformDiscrete(choices)]
    return template, arg_distribution


INPUT_SPACE_FACTORY = {
    # TEXT
    "num_chars": num_chars,
    "num_chars_it": num_chars_instruction_tuned,
    "num_chars_it_1shot": num_chars_instruction_tuned_1shot,
    "num_chars_it_5shot": num_chars_instruction_tuned_5shot,
    "num_chars_it_5shot_system": num_chars_instruction_tuned_5shot_system,  # tested if prompting w chat template was better (it's not)
    "num_chars_it_system": num_chars_instruction_tuned_system,
    "even_odd_it": even_odd_instruction_tuned,
    "even_odd_it_1shot": even_odd_instruction_tuned_1shot,
    "even_odd_it_5shot": even_odd_instruction_tuned_5shot,
    "empty": lambda: [""],
    "empty_it": empty_instruction_tuned,
    "formality_it": formality_it,
    "formality_it_1shot": formality_it_1shot,
    "formality_it_5shot": formality_it_5shot,
    "average_word_length_it": average_word_length_it,
    "average_word_length_it_1shot": average_word_length_it_1shot,
    "average_word_length_it_5shot": average_word_length_it_5shot,
    # IMAGES
    "white_bg_objects": white_bg_objects,
    "white_bg_position_objects": white_bg_position_objects,
    "saturation": saturation,
}

FEEDBACK_FACTORY = {
    "num_chars": ConditionalDistribution(feedback_function=num_chars_dialogue),
    # "num_chars_it_dialogue_llm": ConditionalDistribution(feedback_function=num_chars_dialogue_llm),
    "even_odd": ConditionalDistribution(feedback_function=even_odd_dialogue),
    # "even_odd_it_dialogue_llm": ConditionalDistribution(feedback_function=even_odd_dialogue_llm),
    "formality": ConditionalDistribution(feedback_function=formality_dialogue),
    # "formality_it_dialogue_llm": ConditionalDistribution(feedback_function=formality_dialogue_llm),
    "average_word_length": ConditionalDistribution(
        feedback_function=average_word_length_dialogue
    ),
    # "average_word_length_it_dialogue_llm": ConditionalDistribution(feedback_function=average_word_length_dialogue_llm),
}


def value_num_chars(text: str) -> int | None:
    matches = re.findall(r"Write a string of (\d+) characters", text)
    if matches:
        return int(matches[-1])
    return None


def value_even_odd(text: str) -> bool | None:
    matches = re.findall(r"\b(even|odd)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == "even"
    return None


def value_formality(text: str) -> float | None:
    matches = re.findall(r"\d+\.\d+", text)
    if matches:
        return float(matches[-1])
    return None


def value_average_word_length(text: str) -> float | None:
    matches = re.findall(
        r"\baverage word length is exactly (\d+(?:\.\d+)?) letters\b", text
    )
    if matches:
        return float(matches[-1])
    return None


def value_white_bg_objects(sentence) -> tuple[int | None, str | None]:
    """
    Extract the count (as int) and object name (as string, singular) from sentences like:
    "White background. 7 tables."

    Returns a tuple: (count, object_name_singular) or (None, None) if not found
    """
    p = inflect.engine()
    pattern = r"(\d+)\s+([a-zA-Z]+)"
    match = re.search(pattern, sentence)

    if match:
        count = int(match.group(1))
        object_name = match.group(2)
        object_singular = p.singular_noun(object_name)

        # singular_noun() returns False if the word is already singular
        if object_singular:
            return count, object_singular
        else:
            return count, object_name

    return None, None


def value_white_bg_position_objects(
    sentence, choices=["top left", "top right", "bottom left", "bottom right", "center"]
):
    """
    Extract the position and object name (singular) from sentences like:
    "White background. A table at the top left of the image."

    Args:
        sentence: The sentence to parse
        choices: List of possible position strings (e.g., ["top left", "top right", ...])

    Returns a tuple: (position, object_singular) or (None, None) if not found
    """
    p = inflect.engine()

    # Create a regex pattern from the choices (escape special chars and join with |)
    position_pattern = "|".join(re.escape(choice) for choice in choices)

    # Pattern: "A {object} at the {position} of the image"
    pattern = rf"A\s+([a-zA-Z]+)\s+at\s+the\s+({position_pattern})\s+of\s+the\s+image"

    match = re.search(pattern, sentence, re.IGNORECASE)

    if match:
        object_name = match.group(1)
        position = match.group(2)

        # Convert to singular
        object_singular = p.singular_noun(object_name)
        if not object_singular:  # Already singular
            object_singular = object_name

        return position, object_singular

    return None, None


def value_extract_saturation(sentence):
    """
    Extract the saturation percentage from sentences like:
    "Generate an image with 75.50% saturation."

    Returns the float value or None if not found
    """
    pattern = r"(\d+\.\d+)(?=%\s+saturation)"
    match = re.search(pattern, sentence, re.IGNORECASE)

    if match:
        return float(match.group(1))

    return None


INPUT_VALUE_FACTORY = {
    "num_chars": value_num_chars,
    "even_odd": value_even_odd,
    "formality": value_formality,
    "average_word_length": value_average_word_length,
    "white_bg_objects": value_white_bg_objects,
    "white_bg_position_objects": value_white_bg_position_objects,
    "saturation": value_extract_saturation,
}


def contains_chat_templates(data):
    """
    Check if a data structure contains chat template dictionaries.

    A chat template dictionary has the structure:
    {'role': <string>, 'content': <string>}

    Args:
        data: Any data structure (list, dict, etc.)

    Returns:
        bool: True if chat template dictionaries are found, False otherwise
    """

    def is_chat_template_dict(obj):
        """Check if an object is a chat template dictionary."""
        if not isinstance(obj, dict):
            return False

        # Check if it has 'role' and 'content' keys with string values
        if "role" in obj and "content" in obj:
            if isinstance(obj.get("role"), str) and isinstance(obj.get("content"), str):
                return True

        return False

    def search_recursively(obj):
        """Recursively search through the data structure."""
        if is_chat_template_dict(obj):
            return True

        if isinstance(obj, dict):
            for value in obj.values():
                if search_recursively(value):
                    return True

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if search_recursively(item):
                    return True

        return False

    return search_recursively(data)


def extract_and_flatten_content(data):
    """
    Transform a structure containing chat template dictionaries to only contain
    the 'content' values, and flatten one level of list nesting.

    Args:
        data: Any data structure (list, dict, etc.)

    Returns:
        The same structure with chat template dicts replaced by their 'content' values
        and one level of list nesting flattened
    """

    def transform_recursively(obj, flatten_next=False):
        """Recursively transform the structure."""
        # Check if this is a chat template dict
        if isinstance(obj, dict) and "role" in obj and "content" in obj:
            return obj["content"]

        # Recursively process dictionaries
        if isinstance(obj, dict):
            return {
                key: transform_recursively(value, flatten_next)
                for key, value in obj.items()
            }

        # Recursively process lists
        elif isinstance(obj, list):
            if flatten_next:
                # Flatten one level
                result = []
                for item in obj:
                    transformed = transform_recursively(item, False)
                    if isinstance(transformed, list):
                        result.extend(transformed)
                    else:
                        result.append(transformed)
                return result
            else:
                # Check if we should flatten the next level
                # This happens when we have a list of lists where inner lists contain single items
                transformed = [transform_recursively(item, False) for item in obj]

                # Check if all items are single-element lists
                if all(
                    isinstance(item, list) and len(item) == 1 for item in transformed
                ):
                    return [item[0] for item in transformed]

                return transformed

        # Recursively process tuples
        elif isinstance(obj, tuple):
            return tuple(transform_recursively(item, flatten_next) for item in obj)

        # Return other types as-is
        return obj

    return transform_recursively(data)


def parse_input_strings(task: str, inputs: any) -> any:
    prefix = None
    for prefix in INPUT_VALUE_FACTORY.keys():
        if task.startswith(prefix):
            break
    fn = INPUT_VALUE_FACTORY[prefix]

    def fn_elem(elem):
        o = fn(elem)
        if isinstance(o, (tuple, list)):
            o = o[0]
        return o

    parsed = tree.map_structure(fn_elem, inputs)
    use_template = contains_chat_templates(inputs)
    if use_template:
        parsed = extract_and_flatten_content(parsed)
    return parsed
