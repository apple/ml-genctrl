# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""
Mock models for fast testing without loading real HuggingFace models.

These mocks produce EXACTLY the same output format as real models but are deterministic
and require no model downloads or GPU inference.
"""

import hashlib
import numpy as np
from typing import Iterable
from PIL import Image, ImageDraw


def mock_llm_model(x, u, t=0):
    """
    Mock LLM model matching the exact interface of Model._from_hf_llm.

    Args:
        x: Current state (list of strings or chat histories)
        u: Input (list of strings or chat templates)
        t: Time step

    Returns:
        Tuple of (x_next, x_outs) where:
        - x_next: list of next states (full conversation or prompt+output)
        - x_outs: list of newly generated text only
    """
    # Detect if using chat template format
    is_template = isinstance(u[0], (dict, list))

    if is_template:
        if t == 0:
            # Initialize chat history
            chats = []
            for i, ui in enumerate(u):
                chat_copy = []
                for msg in ui:
                    msg_copy = msg.copy()
                    if msg_copy.get("role") == "user":
                        msg_copy["content"] = x[i] + "\n" + msg_copy["content"]
                    chat_copy.append(msg_copy)
                chats.append(chat_copy)
        else:
            # Continue existing conversation
            chats = [x[i] + [u[i]] for i in range(len(x))]
    else:
        # Simple string concatenation
        prompts = [xi + ui for xi, ui in zip(x, u)] if u is not None else x

    # Generate deterministic outputs
    x_outs = []
    for i in range(len(x)):
        # Create seed from input
        if is_template:
            seed_str = str(chats[i])
        else:
            seed_str = prompts[i]

        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % 10000
        np.random.seed(seed)

        # Generate varying length text
        words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "hello", "world", "test", "data", "output", "response", "result"]
        num_words = np.random.randint(8, 18)
        text = " ".join([words[np.random.randint(len(words))] for _ in range(num_words)])

        x_outs.append(text)

    # Format output
    if is_template:
        # Remove artifacts and format as chat
        x_outs_clean = [x.replace("user", "").strip() for x in x_outs]
        x_outs_chat = [[{"role": "assistant", "content": out}] for out in x_outs_clean]
        x_next = [chats[i] + x_outs_chat[i] for i in range(len(chats))]
        return x_next, x_outs_clean
    else:
        x_next = [prompts[i] + x_outs[i] for i in range(len(prompts))]
        return x_next, x_outs


def mock_t2i_model(x, u, t=0):
    """
    Mock T2I model matching the exact interface of Model._from_hf_t2i.

    Args:
        x: Current state (list of strings)
        u: Input (list of values to concatenate)
        t: Time step

    Returns:
        Tuple of (state_output, images) where:
        - state_output: list of (prompt, image) tuples
        - images: list of PIL Image objects
    """
    # Create prompts
    prompts = [xi + str(ui) for xi, ui in zip(x, u)] if u is not None else x

    # Generate deterministic images
    images = []
    for prompt in prompts:
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 10000
        np.random.seed(seed)

        # Create 256x256 image
        img = Image.new('RGB', (256, 256), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Parse number from prompt (for n_objects task)
        num_objects = 2
        for word in prompt.split():
            if word.isdigit():
                num_objects = int(word)
                break
        num_objects = max(0, min(num_objects, 10))  # Clamp to 0-10

        # Draw objects at deterministic positions
        positions = [(64, 64), (192, 64), (128, 128), (64, 192), (192, 192),
                     (128, 64), (64, 128), (192, 128), (128, 192), (110, 110)]

        for i in range(num_objects):
            x_pos, y_pos = positions[i % len(positions)]
            radius = 20 + np.random.randint(-5, 5)
            color = tuple(np.random.randint(30, 220, 3).tolist())
            draw.ellipse([x_pos-radius, y_pos-radius, x_pos+radius, y_pos+radius],
                        fill=color, outline=(0, 0, 0), width=2)

        images.append(img)

    # Return in expected format
    state_output = [(prompts[i], images[i]) for i in range(len(prompts))]
    return state_output, images


def mock_object_detector(
    hf_model_name: str = "mock-owlv2",
    question: str = None,
    bounds: Iterable = None,
    threshold: float = 0.2,
    **unused_kwargs,
):
    """
    Mock object detector matching output_map_from_hf_detector interface.

    Returns a tuple of (output_map_function, bounds) just like the real detector.
    The output_map function takes PIL Images and returns list of integers (counts).
    """
    if bounds is None:
        bounds = [[0, 20]]

    def output_map(samples: Iterable[Image.Image]) -> list[int]:
        """
        Count objects in images deterministically.
        Returns list of integers.
        """
        results = []
        for sample in samples:
            # Convert image to bytes for deterministic hashing
            import io
            buf = io.BytesIO()
            sample.save(buf, format='PNG')
            img_bytes = buf.getvalue()

            seed = int(hashlib.md5(img_bytes).hexdigest()[:8], 16) % 100
            np.random.seed(seed)

            # Return count between bounds
            if isinstance(bounds, set):
                min_count, max_count = min(bounds), max(bounds)
            elif isinstance(bounds[0], (list, tuple)):
                min_count, max_count = bounds[0][0], bounds[0][1]
            else:
                min_count, max_count = min(bounds), max(bounds)

            count = np.random.randint(min_count, min(max_count + 1, min_count + 8))
            results.append(int(count))

        return results

    return output_map, bounds


def mock_formality_scorer(
    hf_model_name: str = "mock-formality-classifier",
    task: str = "text-classification",
    label: str = "formal",
    device: str = "gpu",
    **unused_kwargs,
):
    """
    Mock formality scorer matching output_map_from_hf_pipeline interface.

    Returns a tuple of (output_map_function, bounds) just like the real pipeline.
    The output_map function takes strings and returns list of floats (scores 0-1).
    """
    bounds = [[0, 1]]

    def output_map(samples: Iterable[str]) -> list[float]:
        """
        Score text formality deterministically.
        Returns list of floats between 0 and 1.
        """
        results = []
        for sample in samples:
            seed = int(hashlib.md5(sample.encode()).hexdigest()[:8], 16) % 1000
            np.random.seed(seed)

            # Base score on deterministic hash
            score = np.random.rand()

            # Adjust based on simple text features for more realistic variation
            if len(sample) > 50:
                score = (score + 0.6) / 2  # Longer text tends more formal
            if any(word in sample.lower() for word in ['the', 'therefore', 'however']):
                score = (score + 0.7) / 2

            # Clip to [0, 1]
            score = float(np.clip(score, 0.0, 1.0))
            results.append(score)

        return results

    return output_map, bounds


# Add mock models to the factory
MOCK_FACTORY = {
    "mock-llm": mock_llm_model,
    "mock-t2i": mock_t2i_model,
}
