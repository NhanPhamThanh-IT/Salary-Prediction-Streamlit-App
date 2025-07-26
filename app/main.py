"""
main.py

Entry point for the Salary Prediction App using Streamlit.

This script initializes and runs the SalaryPredictorApp class, which provides
a web interface for predicting salaries based on years of experience.

Author: Nhan Pham
Email: ptnhanit230104@gmail.com
Date: 2025-07-26
Version: 1.0.0
"""

from SalaryPredictorApp import SalaryPredictorApp

if __name__ == "__main__":
    app = SalaryPredictorApp()
    app.run()
