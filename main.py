from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
from meme_generator import generate_meme_image
from caption_generator import generate_caption
import logging
import os
import base64
import time
import asyncio
from pathlib import Path
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("generated_images", exist_ok=True)

app = FastAPI(
    title="Meme Generator Microservice",
    description="FastAPI microservice for ICP canister integration via HTTPS outcalls",
    version="1.0.0"
)

# CORS for ICP canisters - configure your actual canister URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # For development - restrict in production
        "https://*.ic0.app",  # ICP canisters
        "https://*.icp0.io",  # Alternative ICP domain
    ],
    allow_credentials=False,  # ICP outcalls don't support credentials
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration for ICP integration
ICP_MAX_RESPONSE_SIZE = 1.8 * 1024 * 1024  # 1.8MB (leave buffer for headers)
ICP_TIMEOUT = 25  # 25 seconds (leave buffer for 30s ICP timeout)

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

@app.get("/health")
def health_check():
    """Health check for ICP monitoring."""
    return {
        "status": "healthy", 
        "service": "icp-meme-generator",
        "timestamp": int(time.time()),
        "version": "1.0.0"
    }

@app.get("/generate_meme")
async def generate_meme_for_icp(prompt: str) -> Dict[str, Any]:
    """
    Generate a meme optimized for ICP HTTPS outcalls.
    Returns image as base64 to avoid file serving issues.
    
    Args:
        prompt: The meme generation prompt
        
    Returns:
        Dict containing success status, image data, and metadata
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        if len(prompt) > 500:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Prompt too long (max 500 chars)")
        
        logger.info(f"ICP Meme Generation Request - Prompt: {prompt[:100]}...")
        
        # Set timeout for image generation
        try:
            image_path = await asyncio.wait_for(
                asyncio.to_thread(generate_meme_image, prompt.strip()),
                timeout=ICP_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Image generation timeout")
        
        if not image_path:
            raise HTTPException(status_code=500, detail="Failed to generate image")
        
        # Convert image to base64 for ICP response
        image_base64 = image_to_base64(image_path)
        
        if not image_base64:
            raise HTTPException(status_code=500, detail="Failed to process generated image for ICP")
        
        # Get caption data (this should be fast)
        try:
            caption_data = await asyncio.wait_for(
                asyncio.to_thread(generate_caption, prompt),
                timeout=5  # Quick timeout for caption
            )
        except asyncio.TimeoutError:
            logger.warning("Caption generation timeout, using fallback")
            caption_data = {
                "meme_concept": "Generated meme",
                "top_caption": "Generated meme",
                "bottom_caption": "From AI",
                "middle_caption": None,
                "error": None
            }
        
        # Clean up the generated file
        try:
            os.remove(image_path)
        except Exception as e:
            logger.warning(f"Failed to clean up image file: {e}")
        
        processing_time = time.time() - start_time
        
        response_data = {
            "success": True,
            "message": "Meme generated successfully",
            "data": {
                "prompt": prompt,
                "image_base64": image_base64,
                "image_format": "jpeg",
                "caption_data": caption_data,
                "metadata": {
                    "processing_time": round(processing_time, 2),
                    "timestamp": int(time.time()),
                    "service": "icp-meme-generator"
                }
            }
        }
        
        # Check response size
        response_size = len(str(response_data))
        if response_size > ICP_MAX_RESPONSE_SIZE:
            logger.error(f"Response too large for ICP: {response_size} bytes")
            raise HTTPException(status_code=413, detail="Generated image too large for ICP transport")
        
        logger.info(f"Meme generated successfully in {processing_time:.2f}s, response size: {response_size} bytes")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Meme generation failed after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate_meme")
async def generate_meme_post_for_icp(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST version of meme generation for ICP canisters.
    
    Args:
        request: Dictionary containing prompt and optional parameters
        
    Returns:
        Same as GET version but accepts POST requests
    """
    try:
        prompt = request.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required in request body")
        
        return await generate_meme_for_icp(prompt)
        
    except Exception as e:
        logger.error(f"POST meme generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_caption_only")
async def generate_caption_only(prompt: str) -> Dict[str, Any]:
    """
    Generate only captions without image for faster responses.
    Useful for ICP canisters that want to handle image generation separately.
    
    Args:
        prompt: The meme generation prompt
        
    Returns:
        Dict containing caption data
    """
    try:
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        logger.info(f"Caption-only generation for: {prompt[:100]}...")
        
        # Quick caption generation
        caption_data = await asyncio.wait_for(
            asyncio.to_thread(generate_caption, prompt.strip()),
            timeout=10
        )
        
        return {
            "success": True,
            "message": "Caption generated successfully",
            "data": {
                "prompt": prompt,
                "caption_data": caption_data,
                "timestamp": int(time.time())
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Caption generation timeout")
    except Exception as e:
        logger.error(f"Caption generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    """Root endpoint with ICP integration information."""
    return {
        "service": "Meme Generator Microservice",
        "version": "1.0.0",
        "description": "FastAPI microservice designed for ICP canister integration",
        "endpoints": {
            "/generate_meme": {
                "methods": ["GET", "POST"],
                "description": "Generate meme with image as base64",
                "params": "prompt (string)",
                "response": "base64 encoded image + metadata"
            },
            "/generate_caption_only": {
                "method": "GET",
                "description": "Generate captions only (faster)",
                "params": "prompt (string)",
                "response": "caption data only"
            },
            "/health": {
                "method": "GET", 
                "description": "Health check endpoint"
            }
        },
        "icp_considerations": {
            "max_response_size": f"{ICP_MAX_RESPONSE_SIZE/1024/1024:.1f}MB",
            "timeout_limit": f"{ICP_TIMEOUT}s",
            "image_format": "JPEG base64 encoded",
            "cors_enabled": True
        },
        "usage_example": {
            "curl": "curl 'https://your-service.com/generate_meme?prompt=funny cat meme'",
            "javascript": "fetch('https://your-service.com/generate_meme?prompt=funny cat meme')"
        }
    }

# Error handler for ICP compatibility
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler optimized for ICP outcalls."""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": int(time.time()),
            "service": "icp-meme-generator"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for production deployment
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        workers=1,  # Single worker for resource management
        access_log=True,
        log_level="info"
    )