"""Backward-compatible launcher for the original project entry point.

Run ``python app.py`` for the canonical entry point.
"""

import os

from app import app, create_app


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
