"""
SalaryPredictorApp.py

A Streamlit-based web application for predicting salaries based on years of experience.

This script serves as the entry point for the app. It initializes and runs the 
SalaryPredictorApp class, which loads a trained model and provides a user-friendly 
interface for input and prediction.

Author: Nhan Pham
Email: ptnhanit230104@gmail.com
Date: 2025-07-26
Version: 1.0.0
"""

import streamlit as st
import pickle

class SalaryPredictorApp:
    """
    A Streamlit-based web application to predict salary based on years of experience.
    """

    def __init__(self):
        """Initializes the SalaryPredictorApp by loading the model and setting up the page."""
        self.model = self.load_model()
        self.setup_page()

    def load_model(self):
        """
        Loads a trained machine learning model from a pickle file.

        Returns:
            sklearn.base.BaseEstimator: The trained machine learning model.
        """
        with open('model/model.pkl', 'rb') as file:
            return pickle.load(file)

    def setup_page(self):
        """
        Configures the Streamlit page with a title, icon, and layout.
        """
        st.set_page_config(
            page_title="Salary Prediction App",
            page_icon=":money_with_wings:",
            layout="centered",
        )
        st.markdown(
            "<h1 style='text-align: center; color: #4CAF50;'>Salary Prediction App</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center;'>Predict the salary based on years of experience.</p>",
            unsafe_allow_html=True
        )

    def get_user_input(self):
        """
        Renders a number input widget to get years of experience from the user.

        Returns:
            int: The number of years of experience entered by the user.
        """
        return st.number_input(
            "Enter years of experience:",
            min_value=0,
            max_value=50,
            value=0
        )

    def predict_salary(self, experience):
        """
        Predicts salary based on the number of years of experience.

        Args:
            experience (int): Years of experience.

        Returns:
            float: The predicted salary.
        """
        return self.model.predict([[experience]])[0]

    def run(self):
        """
        Runs the main app logic:
            - Gets user input
            - Predicts salary if button is clicked
            - Displays the result
        """
        experience = self.get_user_input()
        if st.button(label="Predict Salary", use_container_width=True):
            salary = self.predict_salary(experience)
            st.success(
                f"The predicted salary for {experience} years of experience is: ${salary:,.2f}"
            )
