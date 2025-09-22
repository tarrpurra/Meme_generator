#!/usr/bin/env python3
"""
Test script to generate a meme image using the meme_generator module.
Demonstrates the full workflow: prompt -> captions -> image generation.
"""

from meme_generator import generate_meme_image
from caption_generator import generate_caption
import json

def main():
    # Test prompt
    prompt = "A cat sitting at a computer looking confused"

    print(f"Generating meme for prompt: {prompt}")
    print("-" * 50)

    # First, generate captions
    print("Step 1: Generating captions...")
    caption_result = generate_caption(prompt)
    print("Caption result:")
    print(json.dumps(caption_result, indent=2))
    print("-" * 50)

    # Then, generate image (which internally uses the captions)
    print("Step 2: Generating image...")
    image_path = generate_meme_image(caption_result)

    if image_path:
        print(f"Success! Image saved to: {image_path}")
    else:
        print("Failed to generate image.")

if __name__ == "__main__":
    main()