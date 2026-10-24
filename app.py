"""
Entry point for Hugging Face Spaces (and any container deploy).
Run locally: uvicorn app:app --port 7860
Run on Spaces: Dockerfile calls this automatically.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "backend"))

from backend.app import app  # noqa: F401  (uvicorn target)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
