@echo off
cd /d "%~dp0"
echo Serving MediaPipe demo at http://127.0.0.1:8765/
echo Open http://127.0.0.1:8765/ in your browser, then allow the camera.
echo Press Ctrl+C to stop.
python -m http.server 8765
