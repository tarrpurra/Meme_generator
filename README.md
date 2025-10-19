# Meme Generator

A Python microservice for generating memes with professional watermarks using Google Gemini for captions and image creation.

## Features
- Generates meme images from user prompts.
- Uses Google Gemini for AI-powered caption and image generation.
- Automatically applies a professional, logo-only watermark to each meme.
- FastAPI backend for easy integration and HTTP API access.
- Serves generated images via a static `/images` endpoint.

## Project Structure
- `main.py` — FastAPI server, handles meme generation requests.
- `meme_generator.py` — Core logic for generating meme images and captions.
- `util.py` — Utility functions, including watermarking logic.
- `requirements.txt` — Python dependencies.
- `service_account.json` — Credentials for Google Gemini (excluded from git).
- `generated_images/` — Output directory for generated memes.
- `watermark_assets/mementic_log.png` — Logo used for watermarking.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Add your Google Gemini credentials to `service_account.json`.

## Usage
1. Start the FastAPI server:
   ```sh
   python3 main.py
   ```
2. Generate a meme via HTTP API:
   ```sh
   curl "http://localhost:8000/generate_meme?prompt=Your+funny+prompt"
   ```
3. Access generated images at:
   ```
   http://localhost:8000/images/<image_filename>
   ```

## Watermarking
- Every meme is watermarked with a professional logo (no text).
- The watermark is placed at the bottom-right, sized and styled for visibility and aesthetics.

## Security
- `service_account.json` is excluded from git via `.gitignore`.

## License
MIT
