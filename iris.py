# Entrypoint for Streamlit deployment (Streamlit Cloud expects a main module)
# This file simply imports the app module to keep the application logic in app.py.

from app import *  # noqa: F401
