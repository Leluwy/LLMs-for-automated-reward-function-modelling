import torch
import open_clip
from PIL import Image

from typing import Optional, Tuple
import numpy as np

import torch

from torchvision.transforms import (
    Normalize,
    Compose,
    InterpolationMode,
    Resize,
    CenterCrop,
)

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def image_transform(
    image_size: int,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that
        # Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)

    def convert_from_uint8_to_float(image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            return image.to(torch.float32) / 255.0
        else:
            return image

    return Compose(
        [
            convert_from_uint8_to_float,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            normalize,
        ]
    )


def load_openclip_model(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
    """
    Load the OpenCLIP model and tokenizer.

    Args:
        model_name (str): Name of the OpenCLIP model to load.
        pretrained (str): Pretrained weights to use.
        device (str, optional): Device to load the model onto ("cuda" or "cpu").
            Defaults to the best available device.

    Returns:
        tuple: The loaded CLIP model and the tokenizer function.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
    clip_model = clip_model.to(device)
    tokenizer = open_clip.tokenizer.tokenize
    return clip_model, tokenizer


def generate_image_embedding(clip_model, image: str, device: str = None):
    """
    Generate a normalized embedding for a given image.

    Args:
        clip_model: The loaded OpenCLIP model.
        image_path (str): The path to the image to encode.
        device (str, optional): Device to use for tensor computations.
            Defaults to the best available device.

    Returns:
        torch.Tensor: The normalized embedding for the image.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Convert image to a PyTorch tensor (if it's not already)
    if isinstance(image, np.ndarray):  # if image is a numpy array
        image = torch.from_numpy(image.copy())

    # Ensure the image has the shape [C, H, W]
    if image.ndimension() == 3:  # shape [H, W, C]
        image = image.permute(2, 0, 1)  # change shape to [C, H, W]

    # Preprocess the image
    preprocess = image_transform(clip_model.visual.image_size[0])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embeddings = clip_model.encode_image(image_tensor, normalize=True)
    return image_embeddings


def generate_prompt_embedding(clip_model, tokenizer, prompt: str, device: str = None):
    """
    Generate a normalized embedding for a given text prompt.

    Args:
        clip_model: The loaded OpenCLIP model.
        tokenizer: The tokenizer function for OpenCLIP.
        prompt (str): The text prompt to encode.
        device (str, optional): Device to use for tensor computations.
            Defaults to the best available device.

    Returns:
        torch.Tensor: The normalized embedding for the text prompt.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer([prompt]).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(inputs).float()
    normalized_embedding = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return normalized_embedding


def cosine_similarity(x, y):
    """
    Compute the cosine similarity between two tensors.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Cosine similarity between the two tensors.
    """
    similarity = torch.nn.functional.cosine_similarity(x, y)
    return similarity


# Example usage
if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_openclip_model()

    # Generate embedding for a prompt
    prompt = f"""              
            The goal of the game is to move the robotic arm's end-effector to the red target dot in the 3D workspace as quickly,
            efficiently, and accurately as possible.
            """
    embedding = generate_prompt_embedding(model, tokenizer, prompt)

    print("Normalized Text Embedding:", embedding.shape)
