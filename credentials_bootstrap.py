import os, json, base64, io, tempfile

def ensure_google_creds():
    """
    Makes GOOGLE_APPLICATION_CREDENTIALS point to a real file.
    Supports any of these:
      1) GOOGLE_CREDENTIALS_JSON = raw JSON
      2) GOOGLE_CREDENTIALS_JSON = base64(JSON)
      3) GOOGLE_APPLICATION_CREDENTIALS = raw JSON (defensive fix)
      4) GOOGLE_APPLICATION_CREDENTIALS = path to a file (already good)
    """
    ga = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    gcj = os.getenv("GOOGLE_CREDENTIALS_JSON")

    def _write_tmp(json_obj) -> str:
        fd, path = tempfile.mkstemp(prefix="gcp-sa-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(json_obj, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        return path

    # Case 1/2: preferred — use GOOGLE_CREDENTIALS_JSON
    if gcj:
        # try JSON first
        try:
            sa_info = json.loads(gcj)
        except Exception:
            # try base64 -> JSON
            sa_info = json.loads(base64.b64decode(gcj).decode("utf-8"))
        return _write_tmp(sa_info)

    # Case 3: user accidentally pasted JSON into GOOGLE_APPLICATION_CREDENTIALS
    if ga and ga.strip().startswith("{"):
        sa_info = json.loads(ga)
        return _write_tmp(sa_info)

    # Case 4: already a path (do nothing)
    if ga and not ga.strip().startswith("{"):
        if not os.path.exists(ga):
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {ga}")
        return ga

    raise RuntimeError(
        "No credentials found. Set GOOGLE_CREDENTIALS_JSON (raw or base64 JSON) "
        "or GOOGLE_APPLICATION_CREDENTIALS (path or raw JSON)."
    )