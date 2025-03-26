# Loan Prediction System

This project is part of the ShadowFox AI/ML internship and focuses on predicting loan approval status using machine learning techniques.

## Project Overview

The loan prediction system analyzes various features of loan applicants to predict whether their loan application will be approved or rejected. This helps financial institutions make faster and more accurate lending decisions.

## Features

- Data preprocessing and feature engineering
- Model training and evaluation
- Prediction functionality
- Performance metrics visualization

## Project Structure

```
loan_prediction/
├── src/
│   └── loan_prediction.py
├── data/
│   └── loan_data.csv
└── README.md
```

## Setup Instructions

1. Navigate to the project directory:
```bash
cd loan_prediction
```

2. Install required dependencies:
```bash
pip install -r ../requirements.txt
```

3. Run the loan prediction script:
```bash
python src/loan_prediction.py
```

## Model Details

The system uses a Random Forest Classifier with the following features:
- Feature scaling using StandardScaler
- Train-test split (80-20)
- Performance evaluation using accuracy, classification report, and confusion matrix
- Visualization of results using matplotlib and seaborn

## Data Requirements

The input data should be in CSV format with the following columns:
- Applicant features (e.g., income, credit score, employment history)
- Target variable (loan approval status)

## Author

[Kiran Dhanvate](https://github.com/KiranDhanvate) 