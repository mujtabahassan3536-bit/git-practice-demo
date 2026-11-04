# Face Scan Control Center

A local browser dashboard for your existing OpenCV webcam face-detection script, styled similarly to the dark payroll dashboard you shared.

## Requirements
- Python 3.10 or newer
- A webcam connected to your computer

## Run on Windows
1. Extract the ZIP file to a folder.
2. Open Command Prompt or PowerShell in that folder.
3. (Recommended) Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```powershell
   python app.py
   ```
6. Open **http://127.0.0.1:5000** in your browser, then click **Start Camera** and **Run Demo Scan**.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000.

## Camera troubleshooting
- Close other apps that may be using the webcam (Zoom, Meet, Camera, etc.).
- In the dashboard, select camera source 1 or 2 if source 0 does not work.
- Check your operating-system camera permissions.
- Keep the terminal window open while using the dashboard. Press `Ctrl+C` there to stop the web app.

## Important note about the demo results
The provided Python script uses `random.choice()` and `random.randint()` for the gender label, marriage count, and children count. This dashboard keeps that behavior so the interface follows your sample, but clearly labels it as randomized demo output. OpenCV's Haar Cascade detects faces; it does not infer those personal attributes.

The app binds to `127.0.0.1` so it is intended to run on your own computer.
