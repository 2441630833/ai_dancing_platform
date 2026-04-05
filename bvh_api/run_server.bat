@echo off
cd /d "%~dp0"
echo Pose-to-BVH API: http://127.0.0.1:8021/docs
echo POST MP4 to http://127.0.0.1:8021/v1/bvh   (default: bvh_root_local=true)
echo BVH + GIF + joints CSV zip: http://127.0.0.1:8021/v1/bvh_preview.zip
echo Joint positions CSV: http://127.0.0.1:8021/v1/joints.csv
echo Blender tips: BLENDER.md
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8021
