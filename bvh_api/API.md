# BVH API — Interface Guide

Convert an MP4 video to BVH motion capture data using MediaPipe Pose Landmarker.

---

## Setup

**Install dependencies**

```bash
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
Interactive docs (Swagger UI): `http://127.0.0.1:8021/docs`

---

## Common Query Parameters

All `POST` endpoints accept these query parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bvh_root_local` | bool | `true` | Shift skeleton so frame-0 Hips is at origin. Recommended for Blender retargeting. Set `false` for absolute world translation. |

---

## Endpoints

### `GET /health`

Health check.

```
GET http://127.0.0.1:8021/health
```

Response:
```json
{ "status": "ok" }
```

---

### `POST /v1/bvh`

Upload an MP4, get a `.bvh` file back.

```
POST http://127.0.0.1:8021/v1/bvh?bvh_root_local=true
Content-Type: multipart/form-data
Body: upload=<your_file.mp4>
```

Response: `text/plain` — BVH file content (UTF-8), download as `.bvh`

Response headers:

| Header | Description |
|---|---|
| `X-Frames` | Number of motion frames |
| `X-FPS` | Detected video FPS |
| `X-Bvh-Root-Local` | `true` or `false` — confirms which mode was used |

**curl example:**
```bash
curl -X POST "http://127.0.0.1:8021/v1/bvh?bvh_root_local=true" \
  -F "upload=@my_video.mp4" \
  -o output.bvh
```

---

### `POST /v1/bvh_preview.zip`

Upload an MP4, get a ZIP containing:
- `{name}.bvh` — motion capture file
- `{name}_preview.gif` — skeleton overlay animation
- `{name}_joints_world_cm.csv` — joint positions in cm (optional, default included)

```
POST http://127.0.0.1:8021/v1/bvh_preview.zip?bvh_root_local=true&include_joints_csv=true
Content-Type: multipart/form-data
Body: upload=<your_file.mp4>
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bvh_root_local` | bool | `true` | See above |
| `include_joints_csv` | bool | `true` | Include joint positions CSV in the ZIP |

Response: `application/zip`

**curl example:**
```bash
curl -X POST "http://127.0.0.1:8021/v1/bvh_preview.zip" \
  -F "upload=@my_video.mp4" \
  -o output.zip
```

---

### `POST /v1/preview.gif`

Upload an MP4, get only the skeleton overlay GIF (BVH is still computed server-side but not returned).

```
POST http://127.0.0.1:8021/v1/preview.gif
Content-Type: multipart/form-data
Body: upload=<your_file.mp4>
```

Response: `image/gif`

**curl example:**
```bash
curl -X POST "http://127.0.0.1:8021/v1/preview.gif" \
  -F "upload=@my_video.mp4" \
  -o preview.gif
```

---

### `POST /v1/joints.csv`

Upload an MP4, get a wide CSV of smoothed 22-joint world positions in centimeters. Useful for debugging or plotting joint trajectories.

```
POST http://127.0.0.1:8021/v1/joints.csv
Content-Type: multipart/form-data
Body: upload=<your_file.mp4>
```

Response: `text/csv`

CSV columns: `frame`, `time_sec`, then `{JointName}_x_cm`, `{JointName}_y_cm`, `{JointName}_z_cm` for each of the 22 joints.

**curl example:**
```bash
curl -X POST "http://127.0.0.1:8021/v1/joints.csv" \
  -F "upload=@my_video.mp4" \
  -o joints.csv
```

---

### `POST /v1/bvh.json`

Same as `/v1/bvh` but returns JSON with metadata alongside the BVH text.

```
POST http://127.0.0.1:8021/v1/bvh.json
Content-Type: multipart/form-data
Body: upload=<your_file.mp4>
```

Response:
```json
{
  "meta": {
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "frames": 300,
    "frames_read": 300,
    "frames_without_detection": 2,
    "model": "pose_landmarker_lite",
    "bvh_root_local": true
  },
  "bvh": "HIERARCHY\n..."
}
```

---

## Error Codes

| Code | Meaning |
|---|---|
| `400` | Missing filename, wrong file type (must be `.mp4`), or video could not be opened |
| `413` | File too large — max 500 MB |
| `422` | Pose processing failed (e.g. no pose detected in video) |
| `500` | Internal error (e.g. GIF/CSV generation failed) |

---

## Constraints

- Input must be `.mp4` format
- Max file size: **500 MB**
- Max frames processed: **30,000**
- Model used: `pose_landmarker_lite` (auto-downloaded on first run)

---

## Blender Import Tips

See [BLENDER.md](BLENDER.md) for full details. Quick summary:

1. **File → Import → Motion Capture (.bvh)**
2. BVH positions are in **centimeters** — scale the armature by `0.01` if 1 Blender unit = 1 m
3. If the armature lies on its side, rotate **−90° on X** and apply
4. With `bvh_root_local=true` (default), the rig starts near the origin — this is intentional

---

## Python Client Example

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
