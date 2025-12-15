# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import ast
import logging
import string
from typing import Iterable, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

# Import mock models for testing
from genctrl.factory.mock_models import mock_object_detector, mock_formality_scorer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_box_contained(box1, box2):
    """Check if box1 is fully contained within box2"""
    return (
        box1[0] >= box2[0]
        and box1[1] >= box2[1]
        and box1[2] <= box2[2]
        and box1[3] <= box2[3]
    )


def remove_containing_boxes(boxes, scores, labels):
    """Remove boxes that fully contain other boxes (remove the bigger containing boxes)"""
    if len(boxes) <= 1:
        return boxes, scores, labels

    keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue

        for j in range(len(boxes)):
            if i == j or not keep_mask[j]:
                continue

            # If box j is contained in box i, remove box i (the containing/bigger box)
            if is_box_contained(boxes[j], boxes[i]):
                keep_mask[i] = False
                break

    keep_indices = torch.where(keep_mask)[0]

    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]

    if labels is not None and len(labels) > 0:
        keep_list = keep_indices.tolist()
        filtered_labels = [labels[i] for i in keep_list]
    else:
        filtered_labels = None

    return filtered_boxes, filtered_scores, filtered_labels


def nms_area_removal(
    boxes,
    scores,
    labels,
    nms_threshold=0.35,
    max_area: float = 0.7,
    im_size: tuple = (256, 256),
):
    """Apply NMS to remove overlapping detections"""
    if len(boxes) == 0:
        return boxes, scores, labels

    # Convert to tensor if not already
    boxes = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes
    scores = torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores

    # Apply torchvision NMS
    keep_indices = torchvision.ops.nms(boxes, scores, nms_threshold)

    # Filter area
    keep_list = []
    for i in keep_indices:
        box = boxes[i]
        area = (box[2] - box[0]) * (box[3] - box[1]) / im_size[0] / im_size[1]
        if area < max_area:
            keep_list.append(i)

    keep_indices = torch.tensor(keep_list, dtype=torch.int64, device=boxes.device)

    # Filter results
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]

    # Fix for labels - check if labels exist and handle tensor indexing properly
    if labels is not None and len(labels) > 0:
        # Convert keep_indices to list for indexing
        keep_list = keep_indices.tolist()
        filtered_labels = [labels[i] for i in keep_list]
    else:
        filtered_labels = None

    # Remove boxes that are fully contained within other boxes
    filtered_boxes, filtered_scores, filtered_labels = remove_containing_boxes(
        filtered_boxes, filtered_scores, filtered_labels
    )

    return filtered_boxes, filtered_scores, filtered_labels


# ATTRIBUTES ON STRINGS =============================================================
# Attribute evaluators: DISCRETE -> DISCRETE.
def even_odd(samples: Iterable[str]) -> list[bool]:
    def is_integer(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    return [
        int(number) % 2 == 0 if is_integer(number) else "error" for number in samples
    ]


# Attribute evaluators: STRING -> DISCRETE
def num_chars(samples: Iterable[str]) -> list[int]:
    return [len(sample) for sample in samples]


def num_words(samples: Iterable[str]) -> list[int]:
    return [len(sample.split()) for sample in samples]


# STRING -> COARSE-GRAINED CATEGORICAL
def coarse_string_length(samples: Iterable[str]) -> list[str]:
    """
    Categorizes the length of strings into coarse-grained categories.
    """
    categories = []
    for sample in samples:
        length = len(sample)
        if length < 10:
            categories.append("short")
        elif 10 <= length < 50:
            categories.append("medium")
        else:
            categories.append("long")
    return categories


def boolean_expression(samples: Iterable[str], evaluate_to: bool = True) -> list[bool]:
    return [ast.literal_eval(sample) == evaluate_to for sample in samples]


# Attribute evaluators: STRING -> REAL
# This would also include neural scorers
def average_word_length(samples: Iterable[str]) -> list[float]:
    def avg_word_length(s: str) -> float:
        words = [w.strip(string.punctuation) for w in s.split()]
        return float(np.mean([len(w) for w in words]))

    return [avg_word_length(s) for s in samples]


# ATTRIBUTES ON IMAGES ============================================================


def saturation(images: Iterable) -> list[float]:
    """
    Computes the saturation of images, normalized to [0, 1].
    """

    def get_saturation(pil_image):
        # Convert to HSV and normalize
        image_hsv = pil_image.convert("HSV")
        hsv_np = np.array(image_hsv) / 255.0  # shape: (H, W, 3)

        # Saturation: average of S channel
        saturation = np.mean(hsv_np[:, :, 1])
        return saturation

    return [float(get_saturation(image)) for image in images]


def object_position(
    images: Iterable[Image.Image],
    hf_model_name: str = "google/owlv2-base-patch16-ensemble",
    obj: str = "red circle",
    threshold: float = 0.2,
) -> Iterable[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hf_model_name.startswith("google/owlv2"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        processor = Owlv2Processor.from_pretrained(hf_model_name)
        model = Owlv2ForObjectDetection.from_pretrained(
            hf_model_name, use_safetensors=True
        )
    else:
        raise NotImplementedError(f"Model {hf_model_name} not supported.")

    model = model.to(device)

    def object_position_in_image(image, target_object="person"):
        im_size = image.size
        if hf_model_name.startswith("google/owl"):
            inputs = processor(
                text=[
                    target_object,
                ],
                images=image,
                return_tensors="pt",
            )

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)

        if hf_model_name.startswith("google/owl"):
            # OWL-ViT post-processing
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            # Apply NMS to remove overlapping detections
            nms_threshold = 0.5  # Adjust this value as needed, lower is more restrictive (allows small overlap).
            max_area = 0.7  # Adjust this value as needed, removes too large objects that are likely a wrong detection.

            if len(results[0]["scores"]) > 0:
                filtered_boxes, filtered_scores, filtered_labels = nms_area_removal(
                    boxes=results[0]["boxes"],
                    scores=results[0]["scores"],
                    labels=results[0]["labels"],
                    nms_threshold=nms_threshold,
                    max_area=max_area,
                    im_size=im_size,
                )

                if len(filtered_boxes) > 0:
                    # Take largest box
                    areas = []
                    for i, box in enumerate(filtered_boxes):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        areas.append(float(area))
                    max_box = filtered_boxes[np.argmax(areas)]
                    center = 0.5 * torch.tensor(
                        [
                            (max_box[2] + max_box[0]) / im_size[0],
                            (max_box[3] + max_box[1]) / im_size[1],
                        ]
                    )
                    if center[0] < 0.33 and center[1] < 0.33:
                        return "top left"
                    elif center[0] > 0.66 and center[1] < 0.33:
                        return "top right"
                    elif center[0] > 0.66 and center[1] > 0.66:
                        return "bottom right"
                    elif center[0] < 0.33 and center[1] > 0.66:
                        return "bottom left"
                    elif (
                        center[0] >= 0.33
                        and center[0] <= 0.66
                        and center[1] >= 0.33
                        and center[1] <= 0.66
                    ):
                        return "center"
                    else:
                        return "none"
                else:
                    return "none"

            else:
                return "none"
                # print("No detections above threshold")

    results = [object_position_in_image(img, target_object=obj) for img in images]
    return results


# Attribute evaluators: identity
def identity(x: Iterable) -> Iterable:
    return x


OUTPUTS_FACTORY = {
    "even_odd": even_odd,
    "num_chars": num_chars,
    "num_words": num_words,
    "coarse_string_length": coarse_string_length,
    "num_chars_coarse": coarse_string_length,
    "average_word_length": average_word_length,
    "object_position": object_position,
    "saturation": saturation,
}

OUTPUT_SPACE_FACTORY = {
    "none": lambda *args, **kwargs: None,
    "even_odd": lambda *args, **kwargs: set([True, False, "error"]),
    "num_chars": lambda bounds, **kwargs: set(range(bounds[0], bounds[1])),
    "num_words": lambda bounds, **kwargs: set(range(bounds[0], bounds[1])),
    "average_word_length": lambda *args, **kwargs: [
        [0, 45]
    ],  # https://en.wikipedia.org/wiki/Longest_word_in_English 45 letters upper bound
    "coarse_string_length": lambda *args, **kwargs: set(["short", "medium", "long"]),
    "num_chars_coarse": lambda *args, **kwargs: set(["short", "medium", "long"]),
    "objects": lambda bounds, **kwargs: set(range(bounds[0], bounds[1])),
    "object_position": lambda *args, **kwargs: set(
        ["top left", "top right", "bottom left", "bottom right", "center"]
    ),
    "saturation": lambda *args, **kwargs: [[0, 1]],  # Normalized saturation
}


# Evaluations with hf pipeline: https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ObjectDetectionPipeline
def output_map_from_hf_pipeline(
    hf_model_name: str,
    task: str = "text-classification",
    label: str = "formal",
    device: str = "gpu",
) -> callable:
    """
    Returns the output map function based on the Hugging Face model name.

    The output map should take in an Iterable and return an Iterable
    """
    # Use mock if model name starts with "mock-"
    if hf_model_name.startswith("mock-"):
        return mock_formality_scorer(hf_model_name, task, label, device)

    device = (
        0 if device == "gpu" else -1
    )  # Convert 'gpu' to 0 and 'cpu' to -1 for Hugging Face pipeline
    pipe = pipeline(task, model=hf_model_name, device=device, truncation=True)

    def output_map(samples: Iterable[str]) -> Iterable:
        """
        Maps the input samples to their corresponding outputs using the Hugging Face pipeline.
        """
        results = pipe(samples)

        # Get the possible labels
        labels = pipe.model.config.id2label
        assert (
            len(labels) == 2
        )  # TODO: We are only supporting binary classification at the moment
        if label is None:
            target_label = labels[
                0
            ]  # Assuming the first label is the target label (e.g., 'POSITIVE'. Up to user to know what this is)
        else:
            target_label = label
        logger.info(
            f"Using label '{target_label}' as the target label for output mapping."
        )

        return [
            result["score"] if result["label"] == target_label else 1 - result["score"]
            for result in results
        ]

    return output_map, [[0, 1]]


def output_map_from_hf_detector(
    hf_model_name: str = "facebook/detr-resnet-50",
    question: str = None,
    bounds: Iterable = [[0, 1]],
    threshold: float = 0.2,
    **unused_kwargs,
) -> callable:
    """
    Returns an output map using a HF Detector.
    """
    # Use mock if model name starts with "mock-"
    if hf_model_name.startswith("mock-"):
        return mock_object_detector(hf_model_name, question, bounds, threshold, **unused_kwargs)

    if type(bounds) == str:
        bounds = OUTPUT_SPACE_FACTORY[bounds]

    bound_result = make_bound_result(bounds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hf_model_name.startswith("facebook/detr"):
        from transformers import DetrForObjectDetection, DetrImageProcessor

        processor = DetrImageProcessor.from_pretrained(hf_model_name)
        model = DetrForObjectDetection.from_pretrained(hf_model_name)
    elif hf_model_name.startswith("SenseTime/deformable"):
        from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

        processor = AutoImageProcessor.from_pretrained(hf_model_name)
        model = DeformableDetrForObjectDetection.from_pretrained(hf_model_name)
    elif hf_model_name.startswith("google/owlvit"):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        processor = OwlViTProcessor.from_pretrained(hf_model_name)
        model = OwlViTForObjectDetection.from_pretrained(
            hf_model_name, use_safetensors=True
        )
    elif hf_model_name.startswith("google/owlv2"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        processor = Owlv2Processor.from_pretrained(hf_model_name)
        model = Owlv2ForObjectDetection.from_pretrained(
            hf_model_name, use_safetensors=True
        )

    model = model.to(device)

    def count_objects(image_path, target_object="person"):
        image = Image.open(image_path)
        im_size = image.size
        if hf_model_name.startswith("google/owl"):
            inputs = processor(
                text=[
                    question,
                ],
                images=image,
                return_tensors="pt",
            )
        else:
            inputs = processor(images=image, return_tensors="pt")
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)

        if hf_model_name.startswith("google/owl"):
            # OWL-ViT post-processing
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            # Apply NMS to remove overlapping detections
            nms_threshold = 0.5  # Adjust this value as needed, lower is more restrictive (allows small overlap).
            max_area = 0.7  # Adjust this value as needed, removes too large objects that are likely a wrong detection.

            if len(results[0]["scores"]) > 0:
                filtered_boxes, filtered_scores, filtered_labels = nms_area_removal(
                    boxes=results[0]["boxes"],
                    scores=results[0]["scores"],
                    labels=results[0]["labels"],
                    nms_threshold=nms_threshold,
                    max_area=max_area,
                    im_size=im_size,
                )

                # DEBUG Stuff
                # n = np.random.randint(10000)
                # visualize_detections(image_path, filtered_boxes, f"/tmp/some_{n}.png")

                count = len(filtered_scores)
            else:
                count = 0
                # print("No detections above threshold")

        else:
            # DETR post-processing (original code)
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            # Count specific objects by label
            count = 0
            for score, label, box in zip(
                results[0]["scores"], results[0]["labels"], results[0]["boxes"]
            ):
                if model.config.id2label[label.item()] == target_object:
                    count += 1

        return count

    def output_map(samples: Iterable) -> Iterable:
        """
        Maps the samples, which are either images or strings, to the response.
        """
        results = []
        for sample in tqdm(
            samples,
            desc=f"Processing samples with HF Detector [{hf_model_name}] for [{question}]",
        ):
            assert type(sample) == Image.Image
            tmpfile = "/tmp/tempimage.png"
            sample.save(tmpfile)

            count = count_objects(tmpfile, question)
            results.append(count)

        # We should bound the results
        # results = [bound_result(result) for result in results]
        return results

    return output_map, bounds


def output_map_from_hf_vlm(
    hf_model_name: str = "google/gemma-3-4b-it",
    question: str = None,
    bounds: Iterable = [[0, 1]],
    **unused_kwargs,
) -> callable:
    """
    Returns an output map using a HF VLM.
    """
    if type(bounds) == str:
        bounds = OUTPUT_SPACE_FACTORY[bounds]

    bound_result = make_bound_result(bounds)

    pipe = pipeline(
        "image-text-to-text",
        model=hf_model_name,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    def output_map(samples: Iterable) -> Iterable:
        """
        Maps the samples, which are either images or strings, to the response.
        """
        results = []
        for sample in tqdm(
            samples, desc=f"Processing samples with HF VLM [{hf_model_name}]"
        ):
            assert type(sample) == Image.Image
            tmpfile = "/tmp/tempimage.png"
            sample.save(tmpfile)

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmpfile},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            output = pipe(text=messages, max_new_tokens=200)
            result = output[0]["generated_text"][-1]["content"]
            results.append(result)

        # We should bound the results
        # results = [bound_result(result) for result in results]
        return results

    return output_map, bounds


def make_bound_result(bounds: Iterable[Union[str, tuple]]):
    # Detect mode: interval mode if bounds are tuples/lists of length 2
    interval_mode = all(isinstance(b, (tuple, list)) and len(b) == 2 for b in bounds)

    if interval_mode:
        # Normalize intervals (ensure (low, high) with low <= high)
        intervals = [(min(lo, hi), max(lo, hi)) for lo, hi in bounds]

        def bound_result(result: str) -> float:
            try:
                val = float(result)
            except ValueError:
                raise ValueError(f"Expected a numeric value, got {result!r}")

            # Check each interval
            for lo, hi in intervals:
                if lo <= val <= hi:
                    return val

            # Outside all intervals → clip to nearest endpoint
            endpoints = [lo for lo, hi in intervals] + [hi for lo, hi in intervals]
            nearest = min(endpoints, key=lambda x: abs(x - val))
            return nearest

    else:
        # Discrete category mode
        categories = set(bounds)

        def bound_result(result: str) -> str:
            # Case 1: string or other categories
            if result not in categories:
                try:
                    # Case: numerical categories
                    val = float(result)
                    if val < min(categories):
                        return min(categories)
                    elif val > max(categories):
                        return max(categories)
                    else:
                        return val
                except:
                    raise ValueError(
                        f"{result!r} not in allowed categories {categories}"
                    )
            return result

    return bound_result
