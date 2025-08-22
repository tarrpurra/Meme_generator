# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio, os, time, json, hashlib, logging
from datetime import datetime, timedelta

# --- External logic (must return a file path under generated_images/) ---
from meme_generator import generate_meme_image  # your function

# ---------------- Config ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meme-api")

APP_TITLE = "Meme Generator Microservice"
GENERATED_DIR = Path("generated_images")
CACHE_DIR = Path("cache_meta")  # small json entries: filename + created_at
GENERATED_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Frontend / public URL for serving images
# IMPORTANT: set on Railway: BASE_URL=https://memegenerator-production.up.railway.app
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# Prompt-cache TTL (seconds)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # default 6h

# ICP constraints
ICP_MAX_RESPONSE_SIZE = int(float(os.getenv("ICP_MAX_RESPONSE_SIZE_BYTES", str(1.8 * 1024 * 1024))))
ICP_TIMEOUT = int(os.getenv("ICP_TIMEOUT_SECONDS", "25"))  # keep < 30s

# Auto-cleanup config
AUTO_CLEANUP_ENABLED = os.getenv("AUTO_CLEANUP_ENABLED", "true").lower() == "true"
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
CLEANUP_OLDER_THAN_HOURS = int(os.getenv("CLEANUP_OLDER_THAN_HOURS", "48"))

# CORS origins (restrict in prod)
DEFAULT_ORIGINS = [
    "https://*.ic0.app",
    "https://*.icp0.io",
    BASE_URL,  # allow self/base url (useful for local)
]
ALLOW_ORIGINS = [o for o in os.getenv("ALLOW_ORIGINS", "").split(",") if o.strip()] or DEFAULT_ORIGINS

# ---------------- App ----------------
app = FastAPI(title=APP_TITLE, version="1.2.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if "*" not in ALLOW_ORIGINS else ["*"],
    allow_credentials=False,  # ICP outcalls don't use credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images
app.mount("/images", StaticFiles(directory=str(GENERATED_DIR)), name="images")

# Add Cache-Control for images
@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/images/"):
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
    meta = _mem_index.get(key)
    if meta:
        if time.time() - meta["created_at"] < CACHE_TTL_SECONDS and (GENERATED_DIR / meta["filename"]).exists():
            return meta
        _mem_index.pop(key, None)

    p = _meta_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data["created_at"] < CACHE_TTL_SECONDS and (GENERATED_DIR / data["filename"]).exists():
            _mem_index[key] = data
            return data
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

# ---------------- Cleanup (background) ----------------
cleanup_task: Optional[asyncio.Task] = None
last_cleanup_time: Optional[datetime] = None

def cleanup_old_images_sync(older_than_hours: int = CLEANUP_OLDER_THAN_HOURS) -> Dict[str, Any]:
    try:
        if not GENERATED_DIR.exists():
            return {"success": True, "message": "No generated_images directory found"}
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        deleted_files = []
        total_files = 0
        total_size_deleted = 0

        for filename in os.listdir(GENERATED_DIR):
            p = GENERATED_DIR / filename
            if p.is_file():
                total_files += 1
                if p.stat().st_ctime < cutoff_time:
                    try:
                        sz = p.stat().st_size
                        p.unlink()
                        deleted_files.append(filename)
                        total_size_deleted += sz
                    except Exception as e:
                        logger.error(f"Cleanup: failed to delete {filename}: {e}")

        return {
            "success": True,
            "deleted_files": deleted_files,
            "deleted_count": len(deleted_files),
            "total_files_checked": total_files,
            "total_size_deleted_bytes": total_size_deleted,
            "cutoff_hours": older_than_hours,
            "timestamp": int(current_time),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "timestamp": int(time.time())}

async def periodic_cleanup():
    global last_cleanup_time
    logger.info(f"Starting periodic cleanup every {CLEANUP_INTERVAL_HOURS}h")
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)
        result = cleanup_old_images_sync(CLEANUP_OLDER_THAN_HOURS)
        last_cleanup_time = datetime.now()
        if result.get("success"):
            logger.info(f"Cleanup: deleted {result.get('deleted_count', 0)} files")
        else:
            logger.error(f"Cleanup failed: {result.get('error')}")

@app.on_event("startup")
async def on_startup():
    global cleanup_task, last_cleanup_time
    if AUTO_CLEANUP_ENABLED:
        cleanup_task = asyncio.create_task(periodic_cleanup())
        last_cleanup_time = datetime.now()
        # delayed initial cleanup
        async def initial_cleanup():
            await asyncio.sleep(300)
            cleanup_old_images_sync(CLEANUP_OLDER_THAN_HOURS)
        asyncio.create_task(initial_cleanup())

@app.on_event("shutdown")
async def on_shutdown():
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "icp-meme-generator",
        "timestamp": int(time.time()),
        "version": "1.2.0",
        "auto_cleanup": {
            "enabled": AUTO_CLEANUP_ENABLED,
            "interval_hours": CLEANUP_INTERVAL_HOURS,
            "cleanup_older_than_hours": CLEANUP_OLDER_THAN_HOURS,
            "last_cleanup": last_cleanup_time.isoformat() if last_cleanup_time else None,
        },
    }

@app.get("/")
def index():
    return {
        "service": APP_TITLE,
        "version": "1.2.0",
        "base_url": BASE_URL,
        "endpoints": {
            "GET/POST /generate_meme": "Generate meme, returns image URL + metadata",
            "GET /list_generated_images": "List stored images",
            "DELETE /cleanup_old_images?older_than_hours=": "Manual cleanup",
            "GET /cleanup_status": "Auto-cleanup status",
            "POST /trigger_cleanup": "Run cleanup now",
            "GET /health": "Health probe",
        },
        "usage_examples": {
            "curl_get": f"curl '{BASE_URL}/generate_meme?prompt=funny cat meme'",
            "curl_post": f"curl -X POST '{BASE_URL}/generate_meme' -H 'Content-Type: application/json' -d '{{\"prompt\":\"funny cat meme\"}}'",
        },
    }

@app.get("/generate_meme")
async def generate_meme_get(prompt: str) -> Dict[str, Any]:
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    k = _key(prompt)
    # 1) cache
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
                "timestamp": int(time.time()),
                "service": "icp-meme-generator",
            },
        }

    # 2) miss → per-key lock
    lock = _get_lock(k)
    async with lock:
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
                    "timestamp": int(time.time()),
                    "service": "icp-meme-generator",
                },
            }

        # Generate under ICP timeout
        start = time.time()
        try:
            image_path = await asyncio.wait_for(
                asyncio.to_thread(generate_meme_image, prompt),
                timeout=ICP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Image generation timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Image generation returned no file")

        filename = os.path.basename(image_path)
        _write_meta(k, filename)

        processing_time = round(time.time() - start, 2)
        resp = {
            "success": True,
            "data": {
                "prompt": prompt,
                "cached": False,
                "image_url": f"{BASE_URL}/images/{filename}",
                "image_filename": filename,
                "image_format": Path(filename).suffix.lstrip(".").lower() or "png",
                "timestamp": int(time.time()),
                "metadata": {
                    "processing_time": processing_time,
                    "file_size_bytes": os.path.getsize(image_path),
                    "service": "icp-meme-generator",
                },
            },
        }

        # Basic size guard for ICP
        if len(json.dumps(resp)) > ICP_MAX_RESPONSE_SIZE:
            raise HTTPException(status_code=413, detail="Response too large for ICP")

        return resp

@app.post("/generate_meme")
async def generate_meme_post(body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    return await generate_meme_get(prompt)

@app.get("/list_generated_images")
async def list_generated_images():
    if not GENERATED_DIR.exists():
        return {"success": True, "images": [], "message": "No generated_images directory found"}
    files = []
    total_size = 0
    for filename in os.listdir(GENERATED_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            p = GENERATED_DIR / filename
            st = p.stat()
            total_size += st.st_size
            files.append({
                "filename": filename,
                "url": f"{BASE_URL}/images/{filename}",
                "size_bytes": st.st_size,
                "created_time": int(st.st_ctime),
                "age_hours": round((time.time() - st.st_ctime) / 3600, 1),
            })
    return {
        "success": True,
        "images": sorted(files, key=lambda x: x["created_time"], reverse=True),
        "total_count": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

@app.delete("/cleanup_old_images")
async def cleanup_old_images_manual(older_than_hours: int = 24):
    return cleanup_old_images_sync(older_than_hours)

@app.get("/cleanup_status")
async def cleanup_status():
    return {
        "auto_cleanup_enabled": AUTO_CLEANUP_ENABLED,
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "cleanup_older_than_hours": CLEANUP_OLDER_THAN_HOURS,
        "last_cleanup_time": last_cleanup_time.isoformat() if last_cleanup_time else None,
        "next_cleanup_approximate": (
            (last_cleanup_time + timedelta(hours=CLEANUP_INTERVAL_HOURS)).isoformat()
            if last_cleanup_time else "Unknown"
        ),
        "cleanup_task_running": cleanup_task is not None and not cleanup_task.done(),
    }

@app.post("/trigger_cleanup")
async def trigger_cleanup_now():
    result = cleanup_old_images_sync(CLEANUP_OLDER_THAN_HOURS)
    return {"message": "Manual cleanup completed", "result": result}

# ---------------- Error handling ----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc)[:200],
            "timestamp": int(time.time()),
            "service": "icp-meme-generator",
        },
    )

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        access_log=True,
        log_level="info",
        workers=1,
    )
