from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from meme_generator import generate_meme_image
import logging
import os
from fastapi.staticfiles import StaticFiles
import time
import asyncio
from dotenv import load_dotenv
import threading
from datetime import datetime, timedelta

load_dotenv()

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

app.mount("/images", StaticFiles(directory="generated_images"), name="images")

# CORS for ICP canisters - configure your actual canister URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # For development - restrict in production
        "https://*.ic0.app",  # ICP canisters
        "https://*.icp0.io",  # Alternative ICP domain
    ],
    allow_credentials=False,  # ICP outcalls don't support credentials
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

# Configuration for ICP integration
ICP_MAX_RESPONSE_SIZE = 1.8 * 1024 * 1024  # 1.8MB (leave buffer for headers)
ICP_TIMEOUT = 25  # 25 seconds (leave buffer for 30s ICP timeout)

# Base URL configuration - you can set this via environment variable
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# Auto-cleanup configuration
AUTO_CLEANUP_ENABLED = os.getenv("AUTO_CLEANUP_ENABLED", "true").lower() == "true"
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
CLEANUP_OLDER_THAN_HOURS = int(os.getenv("CLEANUP_OLDER_THAN_HOURS", "48"))

# Global variable to track cleanup task
cleanup_task = None
last_cleanup_time = None

def cleanup_old_images_sync(older_than_hours: int = CLEANUP_OLDER_THAN_HOURS) -> Dict[str, Any]:
    """Synchronous version of cleanup for background task."""
    try:
        if not os.path.exists("generated_images"):
            logger.info("No generated_images directory found for cleanup")
            return {"message": "No generated_images directory found"}
       
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)  # Convert hours to seconds
       
        deleted_files = []
        total_files = 0
        total_size_deleted = 0
        
        for filename in os.listdir("generated_images"):
            file_path = os.path.join("generated_images", filename)
            if os.path.isfile(file_path):
                total_files += 1
                file_time = os.path.getctime(file_path)
                if file_time < cutoff_time:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_files.append(filename)
                        total_size_deleted += file_size
                        logger.info(f"Auto-cleanup: Deleted old file: {filename}")
                    except Exception as e:
                        logger.error(f"Auto-cleanup: Failed to delete {filename}: {str(e)}")
        
        result = {
            "success": True,
            "message": f"Auto-cleanup completed",
            "deleted_files": deleted_files,
            "deleted_count": len(deleted_files),
            "total_files_checked": total_files,
            "total_size_deleted_bytes": total_size_deleted,
            "cutoff_hours": older_than_hours,
            "timestamp": int(current_time)
        }
        
        if deleted_files:
            logger.info(f"Auto-cleanup: Deleted {len(deleted_files)} files, freed {total_size_deleted} bytes")
        else:
            logger.info(f"Auto-cleanup: No files older than {older_than_hours} hours found")
            
        return result
    
    except Exception as e:
        logger.error(f"Auto-cleanup failed: {str(e)}")
        return {"success": False, "error": str(e), "timestamp": int(time.time())}

async def periodic_cleanup():
    """Background task that runs cleanup every CLEANUP_INTERVAL_HOURS hours."""
    global last_cleanup_time
    
    logger.info(f"Starting periodic cleanup task (every {CLEANUP_INTERVAL_HOURS} hours)")
    
    while True:
        try:
            # Wait for the specified interval
            await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)  # Convert hours to seconds
            
            logger.info("Running scheduled cleanup...")
            result = cleanup_old_images_sync(CLEANUP_OLDER_THAN_HOURS)
            last_cleanup_time = datetime.now()
            
            if result.get("success"):
                logger.info(f"Scheduled cleanup completed: {result.get('deleted_count', 0)} files deleted")
            else:
                logger.error(f"Scheduled cleanup failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {str(e)}")
            # Continue the loop even if there's an error
            await asyncio.sleep(60)  # Wait 1 minute before retrying

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts."""
    global cleanup_task, last_cleanup_time
    
    if AUTO_CLEANUP_ENABLED:
        logger.info("Starting automatic cleanup background task...")
        cleanup_task = asyncio.create_task(periodic_cleanup())
        last_cleanup_time = datetime.now()
        
        # Run initial cleanup after 5 minutes to clean up any existing old files
        async def initial_cleanup():
            await asyncio.sleep(300)  # Wait 5 minutes
            logger.info("Running initial cleanup...")
            cleanup_old_images_sync(CLEANUP_OLDER_THAN_HOURS)
            
        asyncio.create_task(initial_cleanup())
    else:
        logger.info("Automatic cleanup is disabled")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks when the app shuts down."""
    global cleanup_task
    
    if cleanup_task:
        logger.info("Shutting down cleanup background task...")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled successfully")

@app.get("/health")
def health_check():
    """Health check for ICP monitoring."""
    return {
        "status": "healthy", 
        "service": "icp-meme-generator",
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "auto_cleanup": {
            "enabled": AUTO_CLEANUP_ENABLED,
            "interval_hours": CLEANUP_INTERVAL_HOURS,
            "cleanup_older_than_hours": CLEANUP_OLDER_THAN_HOURS,
            "last_cleanup": last_cleanup_time.isoformat() if last_cleanup_time else None
        }
    }

@app.get("/generate_meme")
async def generate_meme_for_icp(prompt: str) -> Dict[str, Any]:
    """
    Generate a meme optimized for ICP HTTPS outcalls.
    Returns image URL (caption generation is handled internally by meme_generator).
    
    Args:
        prompt: The meme generation prompt
        
    Returns:
        Dict containing success status, image URL, and metadata
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        if len(prompt) > 500:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Prompt too long (max 500 chars)")
        
        logger.info(f"Meme Generation Request - Prompt: {prompt[:100]}...")
        
        # Generate image with timeout (caption generation happens inside generate_meme_image)
        try:
            image_path = await asyncio.wait_for(
                asyncio.to_thread(generate_meme_image, prompt.strip()),
                timeout=ICP_TIMEOUT
            )
            logger.info(f"Generated image path: {image_path}")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Image generation timeout")
        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
        
        if not image_path:
            raise HTTPException(status_code=500, detail="Failed to generate image")
        
        # Check if file actually exists
        if not os.path.exists(image_path):
            logger.error(f"Generated image file not found at: {image_path}")
            raise HTTPException(status_code=500, detail="Generated image file not found")
        
        # Get file size for metadata
        file_size = os.path.getsize(image_path)
        
        # Create image URL (don't delete the file since we're serving it)
        image_filename = os.path.basename(image_path)
        image_url = f"{BASE_URL}/images/{image_filename}"
        
        processing_time = time.time() - start_time
        
        response_data = {
            "success": True,
            "message": "Meme generated successfully",
            "data": {
                "prompt": prompt,
                "image_url": image_url,
                "image_filename": image_filename,
                "image_format": "png",
                "metadata": {
                    "processing_time": round(processing_time, 2),
                    "timestamp": int(time.time()),
                    "file_size_bytes": file_size,
                    "service": "icp-meme-generator"
                }
            }
        }
        
        # Check response size
        response_size = len(str(response_data))
        if response_size > ICP_MAX_RESPONSE_SIZE:
            logger.error(f"Response too large for ICP: {response_size} bytes")
            raise HTTPException(status_code=413, detail="Response too large for ICP transport")
        
        logger.info(f"Meme generated successfully in {processing_time:.2f}s, response size: {response_size} bytes")
        logger.info(f"Image available at: {image_url}")
        
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

@app.get("/list_generated_images")
async def list_generated_images():
    """List all generated images for debugging purposes."""
    try:
        if not os.path.exists("generated_images"):
            return {"images": [], "message": "No generated_images directory found"}
        
        files = []
        total_size = 0
        for filename in os.listdir("generated_images"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join("generated_images", filename)
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                total_size += file_size
                files.append({
                    "filename": filename,
                    "url": f"{BASE_URL}/images/{filename}",
                    "size_bytes": file_size,
                    "created_time": int(file_stats.st_ctime),
                    "age_hours": round((time.time() - file_stats.st_ctime) / 3600, 1)
                })
        
        return {
            "success": True,
            "images": sorted(files, key=lambda x: x["created_time"], reverse=True),
            "total_count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
    
    except Exception as e:
        logger.error(f"Failed to list images: {str(e)}")
        return {"success": False, "error": str(e)}

@app.delete("/cleanup_old_images")
async def cleanup_old_images_manual(older_than_hours: int = 24):
    """Manual cleanup endpoint - Clean up old generated images to save disk space."""
    result = cleanup_old_images_sync(older_than_hours)
    return result

@app.get("/cleanup_status")
async def cleanup_status():
    """Get information about the automatic cleanup system."""
    return {
        "auto_cleanup_enabled": AUTO_CLEANUP_ENABLED,
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "cleanup_older_than_hours": CLEANUP_OLDER_THAN_HOURS,
        "last_cleanup_time": last_cleanup_time.isoformat() if last_cleanup_time else None,
        "next_cleanup_approximate": (last_cleanup_time + timedelta(hours=CLEANUP_INTERVAL_HOURS)).isoformat() if last_cleanup_time else "Unknown",
        "cleanup_task_running": cleanup_task is not None and not cleanup_task.done()
    }

@app.post("/trigger_cleanup")
async def trigger_cleanup_now():
    """Manually trigger a cleanup operation immediately."""
    logger.info("Manual cleanup triggered via API")
    result = cleanup_old_images_sync(CLEANUP_OLDER_THAN_HOURS)
    return {
        "message": "Manual cleanup completed",
        "result": result
    }

@app.get("/")
def root():
    """Root endpoint with service information."""
    return {
        "service": "Meme Generator Microservice",
        "version": "1.0.0",
        "description": "FastAPI microservice for meme generation with integrated captions and auto-cleanup",
        "base_url": BASE_URL,
        "endpoints": {
            "/generate_meme": {
                "methods": ["GET", "POST"],
                "description": "Generate meme with image URL (captions handled internally)",
                "params": "prompt (string)",
                "response": "image URL + metadata"
            },
            "/list_generated_images": {
                "method": "GET",
                "description": "List all generated images with file info"
            },
            "/cleanup_old_images": {
                "method": "DELETE",
                "description": "Manually clean up old generated images",
                "params": "older_than_hours (int, optional, default: 24)"
            },
            "/cleanup_status": {
                "method": "GET",
                "description": "Get auto-cleanup system status"
            },
            "/trigger_cleanup": {
                "method": "POST",
                "description": "Manually trigger cleanup immediately"
            },
            "/health": {
                "method": "GET", 
                "description": "Health check endpoint with cleanup info"
            }
        },
        "features": {
            "integrated_captions": "Caption generation handled in meme_generator.py",
            "image_serving": "Static file serving via /images/ endpoint",
            "automatic_cleanup": f"Auto-cleanup every {CLEANUP_INTERVAL_HOURS}h (files older than {CLEANUP_OLDER_THAN_HOURS}h)",
            "manual_cleanup": "Manual cleanup endpoints available",
            "icp_compatible": "Optimized for ICP HTTPS outcalls"
        },
        "auto_cleanup": {
            "enabled": AUTO_CLEANUP_ENABLED,
            "interval_hours": CLEANUP_INTERVAL_HOURS,
            "cleanup_older_than_hours": CLEANUP_OLDER_THAN_HOURS,
            "last_cleanup": last_cleanup_time.isoformat() if last_cleanup_time else None
        },
        "usage_examples": {
            "curl_get": f"curl '{BASE_URL}/generate_meme?prompt=funny cat meme'",
            "curl_post": f"curl -X POST '{BASE_URL}/generate_meme' -H 'Content-Type: application/json' -d '{{\"prompt\": \"funny cat meme\"}}'",
            "list_images": f"curl '{BASE_URL}/list_generated_images'",
            "manual_cleanup": f"curl -X DELETE '{BASE_URL}/cleanup_old_images?older_than_hours=48'",
            "cleanup_status": f"curl '{BASE_URL}/cleanup_status'",
            "trigger_cleanup": f"curl -X POST '{BASE_URL}/trigger_cleanup'"
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