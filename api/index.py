"""
api/index.py — Vercel serverless entry point for Flask.

Vercel looks for an `app` object in this file (the Python WSGI app).
We simply import and re-export the Flask app from the project root.
"""
import sys
import os

# Add project root to Python path so all local imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app   # noqa: F401  (Vercel picks up `app` automatically)
