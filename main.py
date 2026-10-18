"""
EduPredict Main Entry Point.
Launches the Streamlit application.
"""

import sys
import subprocess

if __name__ == "__main__":
    print("Starting EduPredict Student Performance & Early Warning System...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])
