"""Development entry point for the inflation prediction application."""

import os

from inflation_app import app, create_app


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
