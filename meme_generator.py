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

    # Create a detailed prompt that includes the concept and caption context
    image_prompt = meme_concept
    if top_caption or bottom_caption:
        caption_parts = []
        if top_caption:
            caption_parts.append(f"top text: '{top_caption}'")
        if middle_caption:
            caption_parts.append(f"middle text: '{middle_caption}'")
        if bottom_caption:
            caption_parts.append(f"bottom text: '{bottom_caption}'")
        if caption_parts:
            image_prompt += f". Meme with {' and '.join(caption_parts)}."

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

            print(f"Created output image using model {model_name}")

            try:
                # Add watermark (bottom-right, translucent)
                add_watermark(str(output_path))
            except Exception as wme:
                print(f"Warning: Failed to add watermark: {wme}")

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


def add_watermark(
    image_path: str,
    watermark_path: str = str(Path("watermark_assets") / "mementic_log.png"),
    opacity: float = 0.5,
    margin: int = 16,
    relative_scale: float = 0.18,
):
    """
    Overlay a translucent watermark logo at the bottom-right corner.

    Args:
        image_path (str): Path to the base image to watermark (modified in place)
        watermark_path (str): Path to the watermark image (PNG with transparency preferred)
        opacity (float): 0.0-1.0 opacity multiplier applied to watermark alpha
        margin (int): Pixel margin from the edges
        relative_scale (float): Target watermark width as a fraction of base image width
    """
    try:
        base = Image.open(image_path).convert("RGBA")
        logo = Image.open(watermark_path).convert("RGBA")

        # Scale watermark relative to base width
        target_w = max(1, int(base.width * relative_scale))
        scale_ratio = target_w / float(logo.width)
        target_h = max(1, int(logo.height * scale_ratio))
        logo = logo.resize((target_w, target_h), Image.LANCZOS)

        # Apply opacity while preserving existing alpha
        r, g, b, a = logo.split()
        a = a.point(lambda p: int(p * max(0.0, min(1.0, opacity))))
        logo = Image.merge("RGBA", (r, g, b, a))

        # Position at bottom-right
        x = base.width - logo.width - margin
        y = base.height - logo.height - margin
        x = max(0, x)
        y = max(0, y)

        base.alpha_composite(logo, dest=(x, y))

        # Save back, convert to RGB for broader compatibility
        base.convert("RGB").save(image_path)
        print(f"Successfully watermarked {image_path}")
    except Exception as e:
        print(f"Warning: Failed to add watermark to {image_path}: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise exception, just log warning
