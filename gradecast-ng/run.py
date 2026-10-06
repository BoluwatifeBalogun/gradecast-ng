"""Entry point.

    python run.py                       development server on port 5000
    gunicorn -w 1 --threads 4 run:app   production
"""
import os

from webapp import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
