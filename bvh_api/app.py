"""HTTP API: upload MP4 -> HumanIK-style BVH (MediaPipe Pose Landmarker route 1)."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from pose_pipeline import video_to_bvh

app = FastAPI(
    title="Pose to BVH",
    description=(
        "Offline pipeline: MP4 (fixed camera, single person) -> BVH (HumanIK-style skeleton). "
        "Use /v1/bvh_preview.zip for BVH plus a GIF overlay on the original video for comparison."
    ),
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _safe_stem(mp4_filename: str) -> str:
    stem = Path(mp4_filename).stem
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem) or "motion"


def _safe_bvh_filename(mp4_filename: str) -> str:
    return f"{_safe_stem(mp4_filename)}.bvh"


@app.post(
    "/v1/bvh",
    response_class=Response,
    responses={
        200: {
            "description": "BVH motion capture text (UTF-8); save as .bvh",
            "content": {
                "text/plain": {
                    "schema": {"type": "string", "format": "binary"},
                },
            },
        },
    },
)
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
        out_name = _safe_bvh_filename(upload.filename)
        headers = {
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Frames": str(meta.get("frames", "")),
            "X-FPS": str(meta.get("fps", "")),
        }
        return Response(
            content=bvh.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers=headers,
        )
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


@app.post(
    "/v1/bvh_preview.zip",
    response_class=Response,
    responses={
        200: {
            "description": "ZIP containing .bvh and skeleton-overlay preview .gif",
            "content": {
                "application/zip": {
                    "schema": {"type": "string", "format": "binary"},
                },
            },
        },
    },
)
async def convert_to_bvh_and_gif_zip(upload: UploadFile = File(...)):
    """BVH plus a GIF of the original frames with the detected pose drawn on top (for side-by-side comparison)."""
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
        bvh, meta = video_to_bvh(tmp_path, preview_gif=True)
        gif_bytes = meta.get("preview_gif_bytes")
        if not gif_bytes:
            raise HTTPException(500, "preview GIF was not generated")

        stem = _safe_stem(upload.filename)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{stem}.bvh", bvh.encode("utf-8"))
            zf.writestr(f"{stem}_preview.gif", gif_bytes)

        zip_bytes = buf.getvalue()
        headers = {
            "Content-Disposition": f'attachment; filename="{stem}_bvh_preview.zip"',
            "X-Frames": str(meta.get("frames", "")),
            "X-FPS": str(meta.get("fps", "")),
            "X-Preview-Gif-Frames": str(meta.get("preview_gif_frames", "")),
        }
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers=headers,
        )
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


@app.post(
    "/v1/preview.gif",
    response_class=Response,
    responses={
        200: {
            "description": "Animated GIF: video frames with pose skeleton overlay",
            "content": {"image/gif": {"schema": {"type": "string", "format": "binary"}}},
        },
    },
)
async def convert_to_preview_gif_only(upload: UploadFile = File(...)):
    """Same pose overlay as in the ZIP export; GIF only (BVH is still computed server-side)."""
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
        _bvh, meta = video_to_bvh(tmp_path, preview_gif=True)
        gif_bytes = meta.get("preview_gif_bytes")
        if not gif_bytes:
            raise HTTPException(500, "preview GIF was not generated")

        stem = _safe_stem(upload.filename)
        out = f"{stem}_preview.gif"
        headers = {
            "Content-Disposition": f'attachment; filename="{out}"',
            "X-Frames": str(meta.get("frames", "")),
            "X-FPS": str(meta.get("fps", "")),
        }
        return Response(content=gif_bytes, media_type="image/gif", headers=headers)
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
