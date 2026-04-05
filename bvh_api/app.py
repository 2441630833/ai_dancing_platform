"""HTTP API: upload MP4 -> HumanIK-style BVH (MediaPipe Pose Landmarker route 1)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse

from pose_pipeline import video_to_bvh

app = FastAPI(
    title="Pose to BVH",
    description="Offline pipeline: MP4 (fixed camera, single person) -> BVH (HumanIK-style skeleton).",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/bvh", response_class=PlainTextResponse)
async def convert_to_bvh(upload: UploadFile = File(...)):
    if not upload.filename:
        raise HTTPException(400, "missing filename")
    lower = upload.filename.lower()
    if not lower.endswith(".mp4"):
        raise HTTPException(400, "expected .mp4 upload")

    suffix = Path(upload.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    try:
        content = await upload.read()
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(413, "file too large (max 500MB)")
        tmp.write(content)
        tmp.close()
        bvh, meta = video_to_bvh(tmp_path)
        headers = {
            "X-Frames": str(meta.get("frames", "")),
            "X-FPS": str(meta.get("fps", "")),
        }
        return PlainTextResponse(bvh, media_type="text/plain", headers=headers)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/v1/bvh.json")
async def convert_to_bvh_json(upload: UploadFile = File(...)):
    """Same as /v1/bvh but returns JSON with metadata (body is still large)."""
    if not upload.filename:
        raise HTTPException(400, "missing filename")
    if not upload.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "expected .mp4 upload")

    suffix = Path(upload.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    try:
        content = await upload.read()
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(413, "file too large (max 500MB)")
        tmp.write(content)
        tmp.close()
        bvh, meta = video_to_bvh(tmp_path)
        return JSONResponse({"meta": meta, "bvh": bvh})
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
