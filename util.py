from typing import Dict, Any, Optional
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
try:
    import cairosvg
    SVG_SUPPORT = True
except (ImportError, OSError) as e:
    SVG_SUPPORT = False
    logging.warning(f"SVG support not available: {e}. Will fall back to PNG.")

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

def add_watermark(image_path: str, watermark_image_path: str = "watermark_assets/mementic_log.png", position: str = "bottom-right", opacity: float = 0.8, margin: int = 20) -> str:
    """
    Add professional logo-only watermark to an image.
    """
    try:
        logger.info(f"=== WATERMARK DEBUG START ===")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Watermark path: {watermark_image_path}")
        logger.info(f"Position: {position}, Opacity: {opacity}, Margin: {margin}")
        
        if not os.path.exists(image_path):
            logger.error(f"Source image not found: {image_path}")
            return image_path
        
        if not os.path.exists(watermark_image_path):
            logger.error(f"Watermark logo not found: {watermark_image_path}")
            return image_path
        
        logger.info(f"Both files exist, proceeding with watermarking...")
        
        with Image.open(image_path) as img:
            logger.info(f"Source image size: {img.size}, mode: {img.mode}")
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                logger.info(f"Converted image to RGBA")
            
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            logger.info(f"Created overlay")
            
            # Always use the logo path
            overlay = _create_professional_watermark(overlay, img.size, None, watermark_image_path, position, margin)
            logger.info(f"Professional watermark created")
            
            if opacity < 1.0:
                overlay = _apply_professional_opacity(overlay, opacity)
                logger.info(f"Applied opacity: {opacity}")
            
            watermarked = Image.alpha_composite(img, overlay)
            logger.info(f"Composited watermark with image")
            
            if watermarked.mode == 'RGBA':
                watermarked = watermarked.convert('RGB')
                logger.info(f"Converted final image to RGB")
            
            base_name = os.path.splitext(image_path)[0]
            extension = os.path.splitext(image_path)[1]
            watermarked_path = f"{base_name}_watermarked{extension}"
            
            save_kwargs = {
                'quality': 100,
                'optimize': True,
                'dpi': (300, 300)
            }
            if extension.lower() == '.png':
                save_kwargs.update({'compress_level': 1})
            
            watermarked.save(watermarked_path, **save_kwargs)
            
            logger.info(f"✅ Professional watermark applied successfully: {watermarked_path}")
            logger.info(f"=== WATERMARK DEBUG END ===")
            return watermarked_path
            
    except Exception as e:
        logger.error(f"❌ Error adding watermark: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return image_path

def _create_professional_watermark(overlay, img_size, text, logo_path, position, margin):
    """Create a professional, subtle watermark with SVG support."""
    draw = ImageDraw.Draw(overlay)
    
    logger.info(f"_create_professional_watermark called with logo_path: {logo_path}")
    
    if logo_path and os.path.exists(logo_path):
        try:
            logo = None
            
            logger.info(f"Logo file exists, attempting to load...")
            
            # Check if it's an SVG file
            if logo_path.lower().endswith('.svg'):
                logger.info(f"SVG file detected")
                target_size = _get_optimal_logo_size(img_size, "professional")
                logo = _load_svg_as_image(logo_path, target_size)
            
            if logo is None:
                logger.info(f"Loading as PNG/JPG...")
                # Fallback to regular image loading (PNG, JPG, etc.)
                with Image.open(logo_path) as logo_img:
                    logger.info(f"Logo loaded: size={logo_img.size}, mode={logo_img.mode}")
                    
                    # Slightly bigger for better visibility (10% of image width, max 120px)
                    max_logo_size = min(img_size[0] * 0.10, 120)
                    logo_ratio = min(max_logo_size / logo_img.width, max_logo_size / logo_img.height)
                    new_size = (int(logo_img.width * logo_ratio), int(logo_img.height * logo_ratio))
                    
                    logger.info(f"Resizing logo to: {new_size}")
                    logo = logo_img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Convert to RGBA for transparency
                    if logo.mode != 'RGBA':
                        logo = logo.convert('RGBA')
                        logger.info(f"Converted logo to RGBA")
            
            # Calculate position, avoid overlapping text (move up if bottom text detected)
            logo_x, logo_y = _calculate_position(position, img_size, logo.size, margin)
            logger.info(f"Logo position calculated: ({logo_x}, {logo_y})")
            
            # If meme text is at bottom, move watermark up with extra margin
            if position == 'bottom-right' and text:
                logo_y = max(logo_y - int(img_size[1] * 0.12), margin)
                logger.info(f"Adjusted logo Y position to avoid text: {logo_y}")
            
            # Make logo bright and clearly visible like professional watermarks
            logo = _apply_opacity(logo, 1.0)  # Full opacity for brightness
            logger.info(f"Applied opacity to logo")
            
            # Add drop shadow for visibility
            shadow = Image.new('RGBA', logo.size, (0, 0, 0, 0))
            shadow_offset = 3
            shadow.paste(logo, (shadow_offset, shadow_offset), logo)
            shadow = _apply_opacity(shadow, 0.3)
            
            logger.info(f"Created shadow")
            
            logo_overlay = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
            logo_overlay.paste(shadow, (logo_x, logo_y), shadow)
            logo_overlay.paste(logo, (logo_x, logo_y), logo)
            overlay = Image.alpha_composite(overlay, logo_overlay)
            
            logger.info(f"✅ Logo watermark successfully composited onto overlay")
            
        except Exception as e:
            logger.error(f"❌ Could not add logo watermark: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning(f"❌ Logo path is None or does not exist: {logo_path}")
    
    return overlay

def _create_minimal_watermark(overlay, img_size, text, logo_path, position, margin):
    """Create an ultra-minimal, clean watermark."""
    draw = ImageDraw.Draw(overlay)
    
    if text:
        # Minimal text - tiny and almost invisible, absolutely clean
        font_size = max(9, min(img_size) // 90)  # Even smaller
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SF-Pro-Text-Ultralight.otf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Light.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = _calculate_position(position, img_size, (text_width, text_height), margin)
        
        # Ultra-minimal text - barely visible, perfectly clean
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 20))
    
    return overlay

def _create_branded_watermark(overlay, img_size, text, logo_path, position, margin):
    """Create a branded watermark that's visible but tasteful with SVG support."""
    draw = ImageDraw.Draw(overlay)
    
    if logo_path and os.path.exists(logo_path):
        # Branded logo - slightly larger than professional
        try:
            logo = None
            
            # Check if it's an SVG file
            if logo_path.lower().endswith('.svg'):
                # Get optimal size for branded watermark (larger)
                target_size = _get_optimal_logo_size(img_size, "branded")
                logo = _load_svg_as_image(logo_path, target_size)
                
            if logo is None:
                # Fallback to regular image loading
                with Image.open(logo_path) as logo_img:
                    max_logo_size = min(img_size[0] * 0.06, 90)  # Branded size
                    logo_ratio = min(max_logo_size / logo_img.width, max_logo_size / logo_img.height)
                    new_size = (int(logo_img.width * logo_ratio), int(logo_img.height * logo_ratio))
                    logo = logo_img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    if logo.mode != 'RGBA':
                        logo = logo.convert('RGBA')
            
            logo_x, logo_y = _calculate_position(position, img_size, logo.size, margin)
            
            logo_overlay = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
            logo_overlay.paste(logo, (logo_x, logo_y), logo)
            overlay = Image.alpha_composite(overlay, logo_overlay)
            
        except Exception as e:
            logger.warning(f"Could not add branded logo: {e}")
    
    if text:
        # Branded text - clean and professional, NO BACKGROUND RECTANGLES
        font_size = max(14, min(img_size) // 50)  # Smaller, more subtle
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SF-Pro-Text-Medium.otf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = _calculate_position(position, img_size, (text_width, text_height), margin)
        
        # NO BACKGROUND RECTANGLE - just clean text with subtle outline
        # Very subtle stroke for readability
        stroke_width = 1
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 80), 
                 stroke_width=stroke_width, stroke_fill=(0, 0, 0, 20))
    
    return overlay

def _apply_professional_opacity(overlay, opacity):
    """Apply opacity while maintaining quality."""
    if opacity >= 1.0:
        return overlay
    
    # Create alpha mask
    alpha = overlay.split()[-1]  # Get alpha channel
    alpha = alpha.point(lambda p: int(p * opacity))  # Apply opacity
    overlay.putalpha(alpha)
    return overlay

def _add_image_watermark(overlay, watermark_path, position, margin, img_size):
    """Add high-quality image watermark to overlay."""
    try:
        with Image.open(watermark_path) as watermark_img:
            # Convert to RGBA for transparency support
            if watermark_img.mode != 'RGBA':
                watermark_img = watermark_img.convert('RGBA')
            
            # Resize watermark to be proportional to main image - larger for better visibility
            max_size = min(img_size) // 5  # Increased from //6 to //5 for better visibility
            
            # Maintain aspect ratio while resizing
            watermark_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Calculate position
            x, y = _calculate_position(position, img_size, watermark_img.size, margin)
            
            # Add subtle drop shadow effect for better visibility
            if position in ['bottom-right', 'bottom-left', 'top-right', 'top-left']:
                # Create shadow
                shadow = Image.new('RGBA', watermark_img.size, (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow)
                
                # Draw shadow slightly offset
                shadow_offset = 2
                for i in range(shadow_offset):
                    for j in range(shadow_offset):
                        shadow.paste(watermark_img, (i, j), watermark_img)
                
                # Apply shadow with low opacity
                shadow = _apply_opacity(shadow, 0.3)
                overlay.paste(shadow, (x + shadow_offset, y + shadow_offset), shadow)
            
            # Paste main watermark
            overlay.paste(watermark_img, (x, y), watermark_img)
            
    except Exception as e:
        logger.error(f"Error adding image watermark: {str(e)}")
    
    return overlay

def _calculate_position(position, img_size, watermark_size, margin):
    """Calculate x, y coordinates for watermark placement."""
    img_width, img_height = img_size
    wm_width, wm_height = watermark_size
    
    positions = {
        "top-left": (margin, margin),
        "top-right": (img_width - wm_width - margin, margin),
        "bottom-left": (margin, img_height - wm_height - margin),
        "bottom-right": (img_width - wm_width - margin, img_height - wm_height - margin),
        "center": ((img_width - wm_width) // 2, (img_height - wm_height) // 2)
    }
    
    return positions.get(position, positions["bottom-right"])

def _apply_opacity(overlay, opacity):
    """Apply opacity to an overlay image."""
    # Create a new overlay with adjusted alpha
    alpha = overlay.getchannel('A')
    alpha = alpha.point(lambda p: int(p * opacity))
    overlay.putalpha(alpha)
    return overlay

def create_default_watermark_configs():
    """
    Return a single professional watermark configuration: logo only, bottom-right, clean and visible.
    """
    return {
        "professional_logo_only": {
            "watermark_image_path": "watermark_assets/mementic_log.png",
            "position": "bottom-right",
            "opacity": 1.0,  # Full brightness like professional platforms
            "margin": 20
        }
    }

def _load_svg_as_image(svg_path: str, target_size: tuple) -> Optional[Image.Image]:
    """
    Load SVG file and convert to PIL Image at specified size.
    SVGs provide perfect scaling without quality loss.
    """
    if not SVG_SUPPORT:
        logger.warning("SVG support not available, falling back to PNG")
        return None
    
    try:
        # Convert SVG to PNG at target size with high quality
        png_data = cairosvg.svg2png(
            url=svg_path, 
            output_width=target_size[0],
            output_height=target_size[1],
            background_color="transparent"
        )
        
        # Load PNG data into PIL Image
        img = Image.open(io.BytesIO(png_data))
        
        # Ensure RGBA mode for transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        return img
        
    except Exception as e:
        logger.error(f"Failed to load SVG {svg_path}: {e}")
        return None

def _get_optimal_logo_size(img_size: tuple, style: str = "professional") -> tuple:
    """Calculate optimal logo size based on image dimensions and style."""
    width, height = img_size
    
    if style == "professional":
        # Very subtle - max 4% of image width
        max_size = min(width * 0.04, 60)
    elif style == "minimal": 
        # Ultra small - max 3% of image width
        max_size = min(width * 0.03, 45)
    elif style == "branded":
        # More visible - max 6% of image width  
        max_size = min(width * 0.06, 90)
    else:
        # Default professional
        max_size = min(width * 0.04, 60)
    
    # Return square dimensions for logos
    return (int(max_size), int(max_size))

def _create_gemini_watermark(overlay, img_size, text, logo_path, position, margin):
    """Create a Google Gemini-inspired watermark - clean, visible, and professional."""
    draw = ImageDraw.Draw(overlay)
    
    # Primary focus on image watermark like Google Gemini
    if logo_path and os.path.exists(logo_path):
        try:
            with Image.open(logo_path) as logo_img:
                # Convert to RGBA for transparency support
                if logo_img.mode != 'RGBA':
                    logo_img = logo_img.convert('RGBA')
                
                # Gemini-style sizing - MUCH MORE VISIBLE and prominent
                # Scale based on image size - making it significantly larger for visibility
                base_size = min(img_size) 
                if base_size >= 1000:
                    target_size = 120  # Much larger for big images (was 60)
                elif base_size >= 500:
                    target_size = 80   # Bigger for medium images (was 45)
                else:
                    target_size = 60   # Still visible on small images (was 30)
                
                # Maintain aspect ratio while resizing
                logo_ratio = min(target_size / logo_img.width, target_size / logo_img.height)
                new_size = (int(logo_img.width * logo_ratio), int(logo_img.height * logo_ratio))
                logo = logo_img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Calculate position - Gemini typically uses bottom-right with adequate margin
                logo_x, logo_y = _calculate_position(position, img_size, logo.size, margin)
                
                # Add subtle drop shadow for better visibility (Gemini style)
                shadow = Image.new('RGBA', logo.size, (0, 0, 0, 0))
                shadow_pixels = logo.load()
                if shadow_pixels:
                    for y in range(logo.size[1]):
                        for x in range(logo.size[0]):
                            pixel = shadow_pixels[x, y]
                            if pixel[3] > 0:  # If pixel is not transparent
                                shadow.putpixel((x, y), (0, 0, 0, int(pixel[3] * 0.3)))
                
                # Paste shadow with slight offset
                shadow_offset = 2
                overlay.paste(shadow, (logo_x + shadow_offset, logo_y + shadow_offset), shadow)
                
                # Paste main logo
                overlay.paste(logo, (logo_x, logo_y), logo)
                
        except Exception as e:
            logger.warning(f"Could not add Gemini-style logo watermark: {e}")
    
    # If text is provided, add it near the logo (Gemini sometimes does this)
    if text and logo_path:
        # Much larger text for visibility
        font_size = max(16, min(img_size) // 50)  # Increased from //80 to //50
        
        try:
            # Use system fonts for clean appearance
            font = ImageFont.truetype("/System/Library/Fonts/SF-Pro-Text-Regular.otf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        # Position text to the left of logo
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Adjust position to be left of logo
        if position == "bottom-right":
            text_x = img_size[0] - text_width - target_size - margin - 10
            text_y = img_size[1] - max(text_height, target_size) - margin + (target_size - text_height) // 2
        else:
            text_x, text_y = _calculate_position(position, img_size, (text_width, text_height), margin + target_size + 10)
        
        # Clean text with much better visibility
        draw.text((text_x + 2, text_y + 2), text, font=font, fill=(0, 0, 0, 120))  # Stronger shadow
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 220))  # Much more visible text
    
    elif text and not logo_path:
        # Text-only watermark with Gemini styling - MUCH MORE VISIBLE
        font_size = max(20, min(img_size) // 40)  # Increased from //60 to //40
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SF-Pro-Text-Medium.otf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = _calculate_position(position, img_size, (text_width, text_height), margin)
        
        # Gemini-style text with much better contrast and visibility
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 150))  # Stronger shadow
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 240))  # Much more visible main text
    
    return overlay