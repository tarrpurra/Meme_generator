import os
from pathlib import Path
from google import genai
from google.genai.types import GenerateImagesConfig
from dotenv import load_dotenv
import time

load_dotenv()

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

def generate_art_image(prompt: str, model: str = None) -> str:
    """
    Generate an art image using Google's Imagen API.

    Args:
        prompt (str): The prompt describing the art to generate
        model (str, optional): The Imagen model to use

    Returns:
        str: Path to the generated image file
    """
    try:
        # Get model from parameter or env
        model_name = model or os.getenv("IMAGEN_MODEL", "imagen-3.0-generate-001")

        # Generate image
        image = client.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=GenerateImagesConfig(
                image_size="2K",
            ),
        )

        # Create unique filename
        timestamp = int(time.time())
        output_file = f"art-{timestamp}.png"
        output_path = Path("generated_images") / output_file
        output_path.parent.mkdir(exist_ok=True)

        # Save image
        image.generated_images[0].image.save(str(output_path))

        print(f"Created art image using {len(image.generated_images[0].image.image_bytes)} bytes")

        return str(output_path)

    except Exception as e:
        raise Exception(f"Failed to generate art image: {str(e)}")
