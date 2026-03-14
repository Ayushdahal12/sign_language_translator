"""
Sign Language Translator - Main Application Entry Point
Final Year Project
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ui.interface import run_app

if __name__ == "__main__":
    run_app()
