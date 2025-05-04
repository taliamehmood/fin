# FinFit – Your Financial Health Dashboard

A Streamlit-based Financial Health Dashboard that assesses investor fitness for stock investments using machine learning, themed as a gym/fitness app for finance.

## Features

- Upload Kragle-style financial profile data
- Fetch real-time stock data via Yahoo Finance
- Interactive step-by-step ML pipeline (load, clean, split, train, evaluate)
- Visuals: feature charts, evaluation graphs, prediction results
- Themed as a gym/fitness app for finance (dark mode + red/green tones)
- Button-based navigation + feedback notifications

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- yfinance

## Running the Application

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly yfinance

# Run the application
streamlit run app.py
```

The application will be available at http://localhost:8501

## Dataset Format

The application expects a CSV file with the following columns:
- budget
- risk_tolerance
- investment_horizon
- investment_type
- fitness

Additional columns like income_stability, market_knowledge, debt_ratio, emergency_fund, etc. are supported and will be used as features.

## Deploying on Streamlit Cloud

This application is ready for deployment on Streamlit Cloud. Simply connect your GitHub repository and select the app.py file as the entry point.

## Course Information

- Course: AF3005 – Programming for Finance
- Instructor: Dr. Usama Arshad
- University: FAST-NUCES Islamabad
- Semester: Spring 2025