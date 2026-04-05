@echo off
cd /d "%~dp0"
echo Pose-to-BVH API: http://127.0.0.1:8021/docs
echo POST MP4 to http://127.0.0.1:8021/v1/bvh
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8021
