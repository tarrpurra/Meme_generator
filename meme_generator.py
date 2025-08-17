from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import replicate
import requests
from caption_generator import generate_caption
from dotenv import load_dotenv
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def clean(text):
    """Clean text by removing quotes and newlines"""
    if not text:
        return ""
    return str(text).strip().replace('"', "'").replace("\n", " ")

def generate_meme_image(user_prompt: str):
    """
    Generate a meme image from user prompt.
    First generates captions, then creates image with those captions.
    
    Args:
        user_prompt (str): User's prompt for meme generation
        
    Returns:
        str: Path to generated image file, or None if failed
    """
    print(f"🎯 Processing user prompt: {user_prompt}")
    
    # Ensure the generated_images directory exists
    os.makedirs('generated_images', exist_ok=True)
    
    # Step 1: Generate caption using the caption generator
    print("📝 Generating meme caption...")
    caption_data = generate_caption(user_prompt)
    
    # Check if caption generation failed
    if caption_data.get('error'):
        print(f"❌ Caption generation failed: {caption_data['error']}")
        return None
    
    # Check if we have the required caption data
    if not caption_data.get('meme_concept') or not caption_data.get('top_caption') or not caption_data.get('bottom_caption'):
        print("❌ Invalid caption data received")
        return None
    
    # Step 2: Print the generated caption for user feedback
    print("\n📋 Generated Caption Data:")
    print(f"  Meme Concept: {caption_data['meme_concept']}")
    print(f"  Top Caption: {caption_data['top_caption']}")
    if caption_data.get('middle_caption'):
        print(f"  Middle Caption: {caption_data['middle_caption']}")
    print(f"  Bottom Caption: {caption_data['bottom_caption']}")
    print()
    
    # Step 3: Create the image generation prompt
    image_prompt = create_image_prompt(caption_data)
    
    # Step 4: Generate the image
    image_path = image_generation(image_prompt)
    
    return image_path

def create_image_prompt(caption_data):
    """
    Create a detailed prompt for image generation based on caption data.
    
    Args:
        caption_data (dict): Dictionary containing meme caption data
        
    Returns:
        str: Formatted prompt for image generation
    """
    # Clean the text data
    meme_concept = clean(caption_data['meme_concept'])
    top_caption = clean(caption_data['top_caption'])
    bottom_caption = clean(caption_data['bottom_caption'])
    middle_caption = clean(caption_data.get('middle_caption', ''))
    
    # Build the prompt
    prompt = (
        f"Create a meme image for the following concept: {meme_concept}. "
        f"The image should support these text overlays: "
        f"Top text: '{top_caption}', "
    )
    
    if middle_caption:
        prompt += f"Middle text: '{middle_caption}', "
    
    prompt += (
        f"Bottom text: '{bottom_caption}'. "
        "Make the image clear, high-quality, and suitable for meme format. "
        "The image should be visually engaging and support the humor of the captions. "
        "IMPORTANT: Create a unique, original image that fits the meme concept perfectly."
    )
    
    return prompt

def image_generation(prompt: str):
    """
    Generate a meme image using Gemini as primary method,
    with Replicate Imagen-4 as backup if Gemini fails.
    
    Args:
        prompt (str): The image generation prompt
        
    Returns:
        str: Path to the generated image file, or None if failed
    """
    print("🖼️ Starting image generation...")
    
    # Try Gemini first (as primary)
    image_path = generate_with_gemini(prompt)
    
    if image_path:
        print("✅ Image generated successfully with Gemini")
        return image_path
    else:
        print("⚠️ Gemini failed, trying Replicate Imagen-4 as backup...")
        image_path = generate_with_replicate(prompt)
        if image_path:
            print("✅ Image generated successfully with Replicate Imagen-4 backup")
            print(image_path)
            return image_path
        else:
            print("❌ Both image generation methods failed")
            return None

def generate_with_replicate(prompt):
    """
    Generate image using Replicate Imagen-4
    
    Args:
        prompt (str): Image generation prompt
        
    Returns:
        str: Image path if successful, None if failed
    """
    try:
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            logger.error("REPLICATE_API_TOKEN not found in environment variables")
            return None
        
        client = replicate.Client(api_token=replicate_token)
        
        input_data = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "safety_filter_level": "block_medium_and_above"
        }
        
        print("🔄 Generating image with Replicate Imagen-4...")
        output = client.run("google/imagen-4", input=input_data)
        
        if output:
            try:
                # Handle different types of output from Replicate
                image_url = None
                
                # Method 1: Direct FileOutput object
                if hasattr(output, 'url'):
                    image_url = output.url
                    print(f"📥 Found image URL from FileOutput: {image_url}")
                
                # Method 2: Check if it's iterable (list/generator)
                elif hasattr(output, '__iter__') and not isinstance(output, str):
                    try:
                        output_list = list(output)
                        if output_list:
                            first_item = output_list[0]
                            if hasattr(first_item, 'url'):
                                image_url = first_item.url
                            else:
                                image_url = str(first_item)
                            print(f"📥 Found image URL from list: {image_url}")
                    except Exception as e:
                        logger.warning(f"Could not convert output to list: {e}")
                
                # Method 3: Direct string URL
                else:
                    image_url = str(output)
                    print(f"📥 Using direct URL: {image_url}")
                
                if not image_url:
                    logger.error("Could not extract image URL from output")
                    return None
                
                # Download the image
                print(f"⬇️ Downloading image from: {image_url}")
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # FIXED: Save to generated_images directory
                timestamp = int(time.time())
                image_path = f'generated_images/replicate-generated-image-{timestamp}.png'
                
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"💾 Image saved to: {image_path}")
                
                # Display the image (optional - you might want to comment this out for production)
                try:
                    image = Image.open(image_path)
                    # image.show()  # Commented out for server environment
                except Exception as e:
                    logger.warning(f"Could not display image: {e}")
                
                return image_path
                
            except Exception as e:
                logger.error(f"Error processing Replicate output: {e}")
                logger.info(f"Output type: {type(output)}")
                logger.info(f"Output attributes: {dir(output) if hasattr(output, '__dict__') else 'No attributes'}")
                return None
        else:
            logger.error("No output received from Replicate")
            return None
            
    except Exception as e:
        logger.error(f"Replicate image generation failed: {str(e)}")
        return None

def generate_with_gemini(prompt):
    """
    Generate image using Gemini
    
    Args:
        prompt (str): Image generation prompt
        
    Returns:
        str: Image path if successful, None if failed
    """
    try:
        gemini_api_key = os.getenv('GEMINI_API')
        if not gemini_api_key:
            logger.error("GEMINI_API key not found in environment variables")
            return None
        
        client = genai.Client(api_key=gemini_api_key)
        
        print("🔄 Generating image with Gemini...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print("Gemini response:", part.text)
            elif part.inline_data is not None:
                # Save the image with timestamp to avoid conflicts
                timestamp = int(time.time())
                image_path = f'generated_images/gemini-generated-image-{timestamp}.png'
                
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(image_path)
                
                print(f"💾 Image saved to: {image_path}")
                
                # Display the image (optional - you might want to comment this out for production)
                try:
                    # image.show()  # Commented out for server environment
                    pass
                except Exception as e:
                    logger.warning(f"Could not display image: {e}")
                
                return image_path
        
        logger.error("No image data received from Gemini")
        return None
        
    except Exception as e:
        logger.error(f"Gemini image generation failed: {str(e)}")
        return None

