import base64
import requests
import io
import os
from dotenv import load_dotenv
from PIL import Image
import uuid
load_dotenv()
# You can set this in .env or directly here
HF_API_TOKEN = os.getenv("HF_API_TOKEN") 
CONTROLNET_API_URL = "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_canny"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def transform_image_to_style_api(image_bytes: bytes, prompt: str) -> dict:
    """
    Uses Hugging Face API to transform an image using ControlNet.

    Args:
        image_bytes (bytes): Raw image content
        prompt (str): Style prompt (e.g., "Ghibli style")

    Returns:
        dict: { success, generated_image_path } or error
    """
    try:
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode()

        payload = {
            "inputs": {
                "image": base64_image,
                "prompt": prompt,
            },
            "parameters": {
                "num_inference_steps": 30,
                "guidance_scale": 8.5,
                "controlnet_conditioning_scale": 1.0
            }
        }

        response = requests.post(CONTROLNET_API_URL, headers=HEADERS, json=payload)

        if not response.ok:
            return {"error": f"API call failed: {response.status_code} {response.text}"}

        # Hugging Face might return image bytes or base64
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            result_json = response.json()
            if "error" in result_json:
                return {"error": result_json["error"]}
            if "image" in result_json:
                # decode base64
                image_data = base64.b64decode(result_json["image"])
            else:
                return {"error": "Unexpected response structure"}
        elif "image/" in content_type:
            image_data = response.content
        else:
            return {"error": "Unknown response type from API"}

        # Save to disk
        filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join("generated_images", filename)

        with open(output_path, "wb") as out_file:
            out_file.write(image_data)

        return {
            "success": True,
            "generated_image_path": output_path,
            "prompt_used": prompt
        }

    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
