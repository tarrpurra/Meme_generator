# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional
from meme_generator import generate_meme_image

import os, time, json, asyncio, hashlib
from pathlib import Path

# ---------------- Config ----------------
APP_TITLE = "Meme Generator (Local + Cache)"
BASE_URL = "http://127.0.0.1:8000"
GENERATED_DIR = Path("generated_images")
CACHE_DIR = Path("cache_meta")                  # stores small .json entries
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # 6h default

GENERATED_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ---------------- App ----------------
app = FastAPI(title=APP_TITLE, version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static: we still use StaticFiles, but add a tiny middleware to set Cache-Control
app.mount("/images", StaticFiles(directory=str(GENERATED_DIR)), name="images")

@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/images/"):
        # Allow browsers/CDNs to cache images for 1 day
        resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
    return resp

# ---------------- Simple cache helpers ----------------
_locks: Dict[str, asyncio.Lock] = {}
_mem_index: Dict[str, dict] = {}  # tiny in-memory index of recent cache lookups

def _key(prompt: str) -> str:
    return hashlib.md5(prompt.strip().lower().encode()).hexdigest()

def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"

def _read_meta(key: str) -> Optional[dict]:
    # fast in-memory check first
    meta = _mem_index.get(key)
    if meta: 
        # validate TTL and file existence
        if time.time() - meta["created_at"] < CACHE_TTL_SECONDS and (GENERATED_DIR / meta["filename"]).exists():
            return meta
        _mem_index.pop(key, None)  # stale memory entry -> drop

    p = _meta_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        # validate TTL + file
        if time.time() - data["created_at"] < CACHE_TTL_SECONDS and (GENERATED_DIR / data["filename"]).exists():
            _mem_index[key] = data  # remember
            return data
        # stale -> clean file
        p.unlink(missing_ok=True)
    except Exception:
        p.unlink(missing_ok=True)
    return None

def _write_meta(key: str, filename: str) -> dict:
    data = {"filename": filename, "created_at": time.time()}
    _mem_index[key] = data
    _meta_path(key).write_text(json.dumps(data))
    return data

def _get_lock(key: str) -> asyncio.Lock:
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    return {"service": "local-meme-generator", "cache_ttl_seconds": CACHE_TTL_SECONDS}

@app.get("/generate_meme")
async def generate_meme_get(prompt: str) -> Dict[str, Any]:
    """
    GET /generate_meme?prompt=...
    - Checks cache by prompt
    - If miss: generates, writes meta, returns URL
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    k = _key(prompt)
    # 1) cache check (hot path)
    meta = _read_meta(k)
    if meta:
        filename = meta["filename"]
        return {
            "success": True,
            "data": {
                "prompt": prompt,
                "cached": True,
                "image_url": f"{BASE_URL}/images/{filename}",
                "image_filename": filename,
                "image_format": Path(filename).suffix.lstrip(".").lower() or "png",
                "timestamp": int(time.time())
            }
        }

    # 2) miss -> acquire per-key lock to avoid duplicate work under load
    lock = _get_lock(k)
    async with lock:
        # Double-check cache after acquiring the lock (another task might have filled it)
        meta = _read_meta(k)
        if meta:
            filename = meta["filename"]
            return {
                "success": True,
                "data": {
                    "prompt": prompt,
                    "cached": True,
                    "image_url": f"{BASE_URL}/images/{filename}",
                    "image_filename": filename,
                    "image_format": Path(filename).suffix.lstrip(".").lower() or "png",
                    "timestamp": int(time.time())
                }
            }

        # 3) really generate
        try:
            image_path = generate_meme_image(prompt)  # must return a file path in generated_images/
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Image generation failed")

        filename = os.path.basename(image_path)
        _write_meta(k, filename)

        return {
            "success": True,
            "data": {
                "prompt": prompt,
                "cached": False,
                "image_url": f"{BASE_URL}/images/{filename}",
                "image_filename": filename,
                "image_format": Path(filename).suffix.lstrip(".").lower() or "png",
                "timestamp": int(time.time())
            }
        }

@app.post("/generate_meme")
async def generate_meme_post(body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    return await generate_meme_get(prompt)  # reuse logic

# Global error handler (optional)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Server error", "message": str(exc)[:120]},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=True, log_level="info", workers=1)
