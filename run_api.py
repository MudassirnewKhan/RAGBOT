"""
Local Development Server Runner for RAGBOT REST API.
Run via: python run_api.py
"""
import sys
import os
import uvicorn

# Ensure project root is available on sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"\n🚀 Launching RAGBOT REST API on http://{host}:{port}")
    print(f"📖 Interactive Swagger Docs available at http://127.0.0.1:{port}/docs\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
