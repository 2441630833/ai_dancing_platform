# video2bvh

Convert any MP4 video into BVH motion capture data using MediaPipe Pose Landmarker — no markers, no suits, just a video.

Built on top of [MediaPipe](https://github.com/google/mediapipe), this project provides a local HTTP API that accepts a video file and returns a `.bvh` file ready to import into Blender, MotionBuilder, or any BVH-compatible tool.

---

## How it works

1. Upload an MP4 to the local API server
2. MediaPipe detects 33 pose landmarks per frame
3. Landmarks are mapped to a 22-joint BVH skeleton
4. Smoothing is applied and the BVH file is returned

No GPU required. Runs entirely on-device.

---

## Quick Start

**Install dependencies**

```bash
cd bvh_api
pip install -r requirements.txt
```

**Start the server**

```bash
# Windows
run_server.bat

# or directly
python -m uvicorn app:app --host 127.0.0.1 --port 8021
```

Server runs at `http://127.0.0.1:8021`  
Swagger UI (interactive docs): `http://127.0.0.1:8021/docs`

---

## API Overview

All endpoints accept `multipart/form-data` with an `upload` field containing your `.mp4` file.

| Endpoint | Returns | Description |
|---|---|---|
| `GET /health` | JSON | Health check |
| `POST /v1/bvh` | `.bvh` file | BVH motion capture data |
| `POST /v1/bvh_preview.zip` | `.zip` | BVH + preview GIF + joints CSV |
| `POST /v1/preview.gif` | `.gif` | Skeleton overlay animation |
| `POST /v1/joints.csv` | `.csv` | 22-joint world positions in cm |
| `POST /v1/bvh.json` | JSON | BVH text + metadata |

### Common parameter

All `POST` endpoints accept `?bvh_root_local=true` (default). This shifts the skeleton so the Hips joint starts at the origin on frame 0 — recommended for Blender retargeting.

### Quick example

```bash
curl -X POST "http://127.0.0.1:8021/v1/bvh?bvh_root_local=true" \
  -F "upload=@my_video.mp4" \
  -o output.bvh
```

```python
import requests

with open("my_video.mp4", "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:8021/v1/bvh",
        files={"upload": ("my_video.mp4", f, "video/mp4")},
        params={"bvh_root_local": True},
    )

resp.raise_for_status()
with open("output.bvh", "wb") as out:
    out.write(resp.content)

print("Frames:", resp.headers.get("X-Frames"))
print("FPS:", resp.headers.get("X-FPS"))
```

For the full API reference see [bvh_api/API.md](bvh_api/API.md).

---

## Constraints

- Input must be `.mp4`
- Max file size: 500 MB
- Max frames processed: 30,000
- Model: `pose_landmarker_lite` (auto-downloaded on first run)

---

## Blender Import

1. File → Import → Motion Capture (.bvh)
2. BVH positions are in centimeters — scale the armature by `0.01` if 1 Blender unit = 1 m
3. If the armature lies on its side, rotate −90° on X and apply

See [bvh_api/BLENDER.md](bvh_api/BLENDER.md) for full details.

---

## Browser Demo

A live pose overlay demo runs entirely in the browser using the MediaPipe Tasks Vision WASM bundle — no install needed beyond Python for the local server.

```bash
cd demo
python -m http.server 8765
```

Or on Windows just double-click / run:

```bat
demo\run_demo.bat
```

Then open `http://127.0.0.1:8765/` in your browser and allow camera access. You'll see real-time skeleton tracking overlaid on the mirrored webcam feed.

> Must be served over `http://localhost` or `https://` — camera access is blocked on plain `file://` URLs.

Controls on the page let you toggle the skeleton bones and joint dots independently.

---

## Contributing

Pull requests welcome. Please follow the [contribution guidelines](CONTRIBUTING.md).
