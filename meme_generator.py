import os
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.types import GenerateImagesConfig
from dotenv import load_dotenv
import time
from caption_generator import generate_caption
from PIL import Image, ImageDraw, ImageFont
from credentials_bootstrap import ensure_google_creds

load_dotenv()

# Ensure Google credentials are set up
ensure_google_creds()

# Get project and location from env
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

if not PROJECT or not LOCATION:
    raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in environment")

client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=LOCATION,
)

def generate_meme_image(prompt: str, model: str = None) -> str:
    """
    Generate a meme image using Google's Imagen API.
    First generates captions using the caption generator, then uses the meme_concept for image generation.

    Args:
        prompt (str): The user's prompt for meme generation
        model (str, optional): The Imagen model to use. If None, uses IMAGEN_MODEL from env.

    Returns:
        str: Path to the generated image file
    """
    # First, generate captions
    caption_result = generate_caption(prompt)

    if caption_result.get('error'):
        raise Exception(f"Caption generation failed: {caption_result['error']}")

    # Build image prompt using meme concept and captions
    meme_concept = caption_result.get('meme_concept', '')
    top_caption = caption_result.get('top_caption', '')
    bottom_caption = caption_result.get('bottom_caption', '')
    middle_caption = caption_result.get('middle_caption')

    if not meme_concept:
        raise Exception("No meme concept generated")

    # Create image prompt using only the concept (without captions)
    image_prompt = meme_concept

    # Define available models in order of preference
    available_models = [
        "imagen-4.0-generate-001",
        "imagen-3.0-generate-001"
    ]

    # Get preferred model
    preferred_model = model or os.getenv("IMAGEN_MODEL", "imagen-4.0-generate-001")

    # Ensure preferred model is in the list, add it first if not
    if preferred_model not in available_models:
        available_models.insert(0, preferred_model)
    else:
        # Move preferred to front
        available_models.remove(preferred_model)
        available_models.insert(0, preferred_model)

    last_exception = None
    for model_name in available_models:
        try:
            if model_name == "imagen-4.0-generate-001":
                image = client.models.generate_images(
                    model=model_name,
                    prompt=image_prompt,
                    config=GenerateImagesConfig(
                        image_size="2K",
                    ),
                )
                generated_image = image.generated_images[0].image
            elif model_name == "imagen-3.0-generate-001":
                image = client.models.generate_images(
                    model=model_name,
                    prompt=image_prompt,
                    config=GenerateImagesConfig(),
                )
                generated_image = image.generated_images[0].image
            else:
                continue  # Skip unknown models

            # Create unique filename
            timestamp = int(time.time())
            output_file = f"meme-{timestamp}.png"
            output_path = Path("generated_images") / output_file
            output_path.parent.mkdir(exist_ok=True)

            # Save image
            generated_image.save(str(output_path))

            # Add text overlay
            add_text_overlay(str(output_path), top_caption, bottom_caption, middle_caption)

            print(f"Created output image using model {model_name}")

            return str(output_path)

        except Exception as e:
            last_exception = e
            print(f"Model {model_name} failed: {str(e)}")
            continue

    # If all models failed
    raise Exception(f"Failed to generate image with all available models. Last error: {str(last_exception)}")


def add_text_overlay(image_path: str, top_text: str = "", bottom_text: str = "", middle_text: str = ""):
    """
    Add text overlays to an image to create a complete meme.

    Args:
        image_path (str): Path to the image file
        top_text (str): Text for top of image
        bottom_text (str): Text for bottom of image
        middle_text (str): Text for middle of image
    """
    try:
        # Open image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Get image dimensions
        width, height = img.size

        # Use default font
        font = ImageFont.load_default()

        # Simple text drawing without wrapping for now
        # Add top text
        if top_text:
            text = top_text.upper()
            # Simple centering
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = 10
            # White text with black outline
            for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + offset[0], y + offset[1]), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill="white")

        # Add bottom text
        if bottom_text:
            text = bottom_text.upper()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = height - text_height - 10
            for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + offset[0], y + offset[1]), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill="white")

        # Add middle text
        if middle_text:
            text = middle_text.upper()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + offset[0], y + offset[1]), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill="white")

        # Save the image with text
        img.save(image_path)
        print(f"Successfully added text overlay to {image_path}")

    except Exception as e:
        print(f"Warning: Failed to add text overlay: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise exception, just log warning



