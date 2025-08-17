from typing import Dict, Any, Optional
import logging
from pathlib import Path
from PIL import Image
import io
import base64
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def image_to_base64(image_path: str, max_size_mb: float = 1.5) -> Optional[str]:
    """
    Convert image to base64 string with size optimization for ICP.
    
    Args:
        image_path: Path to the image file
        max_size_mb: Maximum size in MB for the base64 string
        
    Returns:
        Base64 encoded image string or None if too large
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        # Open and potentially compress image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Start with original size
            quality = 85
            width, height = img.size
            
            while True:
                # Create a copy for this iteration
                temp_img = img.copy()
                
                # Resize if too large
                if width > 1024 or height > 1024:
                    temp_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                img_bytes = io.BytesIO()
                temp_img.save(img_bytes, format='JPEG', quality=quality, optimize=True)
                img_bytes.seek(0)
                
                # Check size
                size_mb = len(img_bytes.getvalue()) / (1024 * 1024)
                
                if size_mb <= max_size_mb or quality <= 20:
                    # Encode to base64
                    b64_string = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                    logger.info(f"Image converted to base64: {size_mb:.2f}MB, quality: {quality}")
                    return b64_string
                
                # Reduce quality for next iteration
                quality -= 15
                if quality <= 20:
                    # Try reducing dimensions
                    width = int(width * 0.8)
                    height = int(height * 0.8)
                    quality = 85
                    
                    if width < 256 or height < 256:
                        logger.error("Cannot compress image small enough for ICP")
                        return None
        
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None