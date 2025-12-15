# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional

import torch
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    LCMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid annoying print statements
from diffusers.utils import logging as diffuser_logging

diffuser_logging.disable_progress_bar()

from genctrl.utils.utils import prettify_name


class TextToImageGenerator:
    """
    A unified interface for text-to-image generation with multiple models.
    Supports FLUX.1-schnell, Stable Diffusion XL, and DMD2.
    """

    SUPPORTED_MODELS = {
        "black-forest-labs/FLUX.1-schnell": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "pipeline_class": FluxPipeline,
            "default_steps": 4,
            "supported_steps": [1, 4],
            "guidance_scale": 0.0,
            "max_sequence_length": 256,
        },
        "black-forest-labs/FLUX.1-dev": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "pipeline_class": FluxPipeline,
            "default_steps": 50,
            "supported_steps": [1, 4],
            "guidance_scale": 0.0,
            "max_sequence_length": 256,
        },
        "stabilityai/stable-diffusion-xl-base-1.0": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "pipeline_class": StableDiffusionXLPipeline,
            "default_steps": 40,
            "supported_steps": list(range(1, 101)),
            "guidance_scale": 7.5,
            "max_sequence_length": 77,
        },
        "tianweiy/DMD2": {
            "model_id": "tianweiy/DMD2",
            "pipeline_class": DiffusionPipeline,
            "default_steps": 4,
            "supported_steps": [1, 4],
            "guidance_scale": 0.0,
            "max_sequence_length": 77,
            "timesteps": [999, 749, 499, 249],
        },
    }

    def __init__(
        self,
        model_name: str = "flux",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
    ):
        """
        Initialize the text-to-image generator.

        Args:
            model_name: Name of the model to use ("flux", "sdxl", "dmd2")
            device: Device to run on ("auto", "cuda", "mps", "cpu")
            cache_dir: Custom directory for Hugging Face cache (optional)
            num_inference_steps: Number of inference steps to use (uses model default if None)
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. Choose from {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self.cache_dir = cache_dir

        # Set inference steps (use model default if not specified)
        if num_inference_steps is None:
            self.num_inference_steps = self.model_config["default_steps"]
        else:
            self.num_inference_steps = num_inference_steps

        # Validate steps for model
        if self.num_inference_steps not in self.model_config["supported_steps"]:
            logger.warning(
                f"Warning: {self.num_inference_steps} steps may not be optimal for {self.model_name}"
            )
            logger.warning(f"Recommended steps: {self.model_config['supported_steps']}")

        # Set device with MPS support
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.pipeline = None
        self.current_dmd2_steps = None  # Track which DMD2 model is loaded
        self._load_model(self.num_inference_steps)

    def _load_model(self, num_inference_steps: Optional[int] = None):
        """Load the specified model pipeline."""
        logger.info(f"Loading {self.model_name} model (device={self.device})...")

        model_id = self.model_config["model_id"]

        try:
            if "flux" in self.model_name.lower():
                dtype = (
                    torch.bfloat16 if self.device in ["cuda", "mps"] else torch.float32
                )
                self.pipeline = FluxPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, cache_dir=self.cache_dir
                ).to(self.device)

            elif "stable-diffusion" in self.model_name.lower():
                dtype = torch.float16 if self.device in ["cuda"] else torch.float32
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if self.device in ["cuda", "mps"] else None,
                    cache_dir=self.cache_dir,
                )

            elif "dmd2" in self.model_name.lower():
                # Load DMD2 according to official documentation
                from diffusers import UNet2DConditionModel
                from huggingface_hub import hf_hub_download

                base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                repo_name = "tianweiy/DMD2"

                # Choose the appropriate checkpoint based on inference steps
                if num_inference_steps == 1:
                    ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
                else:
                    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

                # Load the UNet configuration from base model (using proper method)
                unet = UNet2DConditionModel.from_config(
                    base_model_id, subfolder="unet"
                ).to("cuda", torch.float16)
                unet.load_state_dict(
                    torch.load(
                        hf_hub_download(repo_name, ckpt_name), map_location="cuda"
                    )
                )
                # For MPS, use float32 even with fp16 weights for compatibility
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                unet = unet.to(self.device, dtype)

                # Download and load DMD2 UNet weights
                unet_path = hf_hub_download(
                    repo_id=repo_name, filename=ckpt_name, cache_dir=self.cache_dir
                )
                unet.load_state_dict(torch.load(unet_path, map_location=self.device))

                # Create pipeline with DMD2 UNet
                # For MPS, use float32 even with fp16 weights for compatibility
                pipeline_dtype = torch.float16
                self.pipeline = DiffusionPipeline.from_pretrained(
                    base_model_id,
                    unet=unet,
                    torch_dtype=pipeline_dtype,
                    variant="fp16",
                    cache_dir=self.cache_dir,
                ).to("cuda")

                # Set LCMScheduler for DMD2
                self.pipeline.scheduler = LCMScheduler.from_config(
                    self.pipeline.scheduler.config
                )

                # Track which DMD2 model is loaded
                self.current_dmd2_steps = (
                    num_inference_steps if num_inference_steps else 4
                )

            self.pipeline.to(device=self.device, dtype=dtype)

            # Enable model CPU offload for FLUX (recommended in documentation)
            if self.model_name == "flux" and hasattr(
                self.pipeline, "enable_model_cpu_offload"
            ):
                self.pipeline.enable_model_cpu_offload()

            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()

            # Enable CPU offloading if on limited GPU memory (not needed for MPS)
            if (
                self.device == "cuda"
                and torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3
            ):  # Less than 16GB
                if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                    self.pipeline.enable_sequential_cpu_offload()

            logger.info(f"Model loaded successfully on {self.device}")

            self.pipeline.set_progress_bar_config(disable=True)

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_images(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        gen_batch_size: int = 1,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: int = 256,
        width: int = 256,
        output_dir: str = "generated_images",
    ) -> List[List[Image.Image]]:
        """
        Generate images from text prompts.

        Args:
            prompts: List of text prompts
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for reproducibility (applied to all images)
            guidance_scale: Guidance scale for generation
            height: Image height
            width: Image width
            output_dir: Directory to save images

        Returns:
            List of lists containing PIL Images (one list per prompt)
        """
        if not self.pipeline:
            raise RuntimeError(
                "Model not loaded. Please initialize the generator first."
            )

        assert not (num_images_per_prompt != 1 and seed is not None), (
            "If seed is set, only 1 repetition per prompt should be generated."
        )

        # Use the inference steps set during initialization
        num_inference_steps = self.num_inference_steps

        if guidance_scale is None:
            guidance_scale = self.model_config["guidance_scale"]

        # Set seed if provided - create generator once and reuse for all images
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        def batchify(
            prompts_list: List[str], batch_size: int
        ) -> Generator[List[str], None, None]:
            """
            Yield successive batches of prompts of size `batch_size`.
            """
            for i in range(0, len(prompts_list), batch_size):
                batch = prompts_list[i : i + batch_size]
                if batch:  # ensure not empty
                    yield batch

        counter = 0
        all_images = []
        logger.info(
            f"Generating from {len(prompts)} prompts with batch size = {gen_batch_size}."
        )
        logger.info(f"Generation config:\n num_inference_steps={num_inference_steps}")
        for batch_i, prompts_batch in enumerate(
            batchify(prompts_list=prompts, batch_size=gen_batch_size)
        ):
            # logger.info(f"Generating {num_images_per_prompt} images for prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            prompt_images = []

            for j in range(num_images_per_prompt):
                try:
                    # Set seed if provided - create generator every time we call the pipeline
                    if seed is not None:
                        generator = torch.Generator(device=self.device).manual_seed(
                            seed
                        )
                    else:
                        generator = None

                    # Generate image
                    start_time = time.time()

                    # Prepare generation parameters
                    gen_params = {
                        "prompt": prompts_batch,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "generator": generator,
                        # "guidance_scale": guidance_scale
                    }

                    # Special handling for FLUX
                    if "flux" in self.model_name.lower():
                        gen_params["max_sequence_length"] = 256

                    # Special handling for DMD2
                    if "dmd2" in self.model_name.lower():
                        # Set specific timesteps for DMD2
                        if num_inference_steps == 4:
                            gen_params["timesteps"] = [999, 749, 499, 249]
                        elif num_inference_steps == 1:
                            gen_params["timesteps"] = [399]
                        gen_params["height"] = 1024
                        gen_params["width"] = 1024

                        # Special handling for DMD2
                    if "stabilityai" in self.model_name.lower():
                        gen_params["height"] = 1024
                        gen_params["width"] = 1024

                    # Generate image
                    result = self.pipeline(**gen_params)
                    generation_time = time.time() - start_time
                    logger.info(
                        f"  Batch {batch_i}, image repetition {j} generated in {generation_time:.2f}s"
                    )

                    for img_id, image in enumerate(result.images):
                        # Save image
                        model_str = prettify_name(self.model_name)
                        filename = f"{model_str}_prompt_{batch_i:04d}_image_{img_id:04d}_seed_{seed if seed is not None else 'random'}.png"
                        filepath = os.path.join(output_dir, filename)
                        image.save(filepath)
                        logger.info(f"  Saved image {filepath}")
                        prompt_images.append(image)
                        counter += 1

                except Exception as e:
                    logger.warning(
                        f"Error generating image {j} for prompt batch {batch_i}: {e}"
                    )
                    continue

            all_images.extend(prompt_images)

        return all_images

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all supported models and their configurations."""
        return self.SUPPORTED_MODELS

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        return {
            "name": self.model_name,
            "model_id": self.model_config["model_id"],
            "device": self.device,
            "default_steps": self.model_config["default_steps"],
            "supported_steps": self.model_config["supported_steps"],
            "guidance_scale": self.model_config["guidance_scale"],
            "current_inference_steps": self.num_inference_steps,
        }

    def set_inference_steps(self, num_inference_steps: int):
        """
        Change the number of inference steps.

        Args:
            num_inference_steps: New number of inference steps

        Note:
            For DMD2, this will reload the model with the appropriate checkpoint.
        """
        # Validate steps for model
        if num_inference_steps not in self.model_config["supported_steps"]:
            print(
                f"Warning: {num_inference_steps} steps may not be optimal for {self.model_name}"
            )
            print(f"Recommended steps: {self.model_config['supported_steps']}")

        # For DMD2, check if we need to reload the model with different steps
        if "dmd2" in self.model_name.lower() and hasattr(self, "current_dmd2_steps"):
            if self.current_dmd2_steps != num_inference_steps:
                print(f"Reloading DMD2 model for {num_inference_steps} steps...")
                self._load_model(num_inference_steps)
                self.current_dmd2_steps = num_inference_steps

        self.num_inference_steps = num_inference_steps
        logger.info(f"Inference steps set to {num_inference_steps}")


# Convenience function for quick usage
def generate_images_simple(
    prompts: List[str],
    model: str = "flux",
    num_images_per_prompt: int = 1,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[List[Image.Image]]:
    """
    Simple function to generate images quickly.

    Args:
        prompts: List of text prompts
        model: Model name ("flux", "sdxl", "dmd2")
        num_images_per_prompt: Number of images per prompt
        seed: Random seed for reproducibility
        steps: Number of inference steps
        cache_dir: Custom directory for Hugging Face cache (optional)

    Returns:
        List of lists containing PIL Images
    """
    generator = TextToImageGenerator(
        model, cache_dir=cache_dir, num_inference_steps=steps
    )
    return generator.generate_images(
        prompts=prompts, num_images_per_prompt=num_images_per_prompt, seed=seed
    )


if __name__ == "__main__":
    cache_dir = "/mnt/data"
    steps = 4
    t2im = TextToImageGenerator(
        "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir=cache_dir,
        num_inference_steps=steps,
    )
    t2im.generate_images(
        prompts=["A cat on a tree", "A cat on a tree"], num_images_per_prompt=2, seed=42
    )
    t2im.generate_images(prompts=["A cat on a tree"], num_images_per_prompt=1, seed=42)
