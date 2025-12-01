# 🏥 Hospital Length of Stay Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue.svg)](https://www.kaggle.com/princepanthi)

A machine learning project that predicts hospital length of stay using patient health metrics with OLS regression analysis.

## 📊 Project Overview

This project analyzes 30,000 patient records to identify key factors influencing hospital stay duration. The model uses Ordinary Least Squares (OLS) regression to predict length of stay based on six health metrics.

### Key Features:
- **Predictors**: Age, Glucose, Blood Pressure, BMI, Cholesterol, HbA1c
- **Target**: Length of stay (days)
- **Algorithm**: OLS Regression with statistical validation
- **Dataset**: 30,000 synthetic patient records
- **Interpretation**: Full coefficient analysis and statistical significance

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Princeee07/Hospital-Length-of-stay.git
cd Hospital-Length-of-stay

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
