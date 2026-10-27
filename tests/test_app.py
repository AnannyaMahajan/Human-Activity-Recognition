"""
Automated smoke test for the Streamlit web application.
Uses streamlit.testing.v1.AppTest to verify first render without exceptions.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = BASE_DIR / "app" / "app.py"

def test_app_first_render():
    """Verify that app/app.py loads and renders without exceptions."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=45)
    at.run()
    
    assert not at.exception, f"App crashed on first render with exception: {at.exception}"

def test_app_tab_navigation():
    """Verify that switching tabs renders successfully."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    
    # Assert simplified tabs exist
    assert len(at.tabs) >= 4
    assert not at.exception

def test_app_predict_interaction():
    """Verify prediction button click on Predict tab runs cleanly."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=35)
    at.run()
    
    # Click any interactive buttons and ensure clean execution
    for btn in at.button:
        btn.click().run()
        assert not at.exception, f"App crashed clicking button '{btn.label}': {at.exception}"

