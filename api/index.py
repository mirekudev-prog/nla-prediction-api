"""
Vercel Serverless Function Entry Point for Ghana NLA 5/90 Prediction API.
This wraps the FastAPI app for Vercel's Python runtime.
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

# Import the FastAPI app
from app import app

# Vercel expects the app to be available as `app`
# For ASGI compatibility with Vercel's Python runtime
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    # Fallback for local testing
    handler = app

# Export for Vercel
__all__ = ["app", "handler"]