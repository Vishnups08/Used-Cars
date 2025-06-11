# Used Car Price Prediction

## Problem Statement
Implement data warehouse/data mining models to analyze selling price of used cars. This project aims to build and compare different machine learning models to predict the selling price of used cars based on various features such as brand, model year, mileage, engine specifications, and more.

## Overview
This project is a Streamlit-based web application that provides an interactive interface for analyzing used car data and predicting car prices using various machine learning models. The application includes data visualization, model training, and price prediction capabilities.

## Features

### 1. Data Overview
- Basic dataset information and statistics
- Interactive visualizations:
  - Price distribution
  - Top car brands distribution
  - Average price by brand
  - Price vs. mileage analysis
  - Price trends by model year

### 2. Data Preprocessing
- Automated data cleaning and preprocessing
- Feature selection capabilities
- Missing value handling
- Correlation analysis
- Data transformation pipeline

### 3. Model Training & Evaluation
- Multiple model options:
  - Linear Regression
  - Random Forest
  - XGBoost
- Performance metrics:
  - RMSE (Root Mean Square Error)
  - R² Score
  - MAE (Mean Absolute Error)
- Model comparison visualizations
- Feature importance analysis

### 4. Price Prediction
- Interactive form for car details input
- Real-time price prediction
- Comparison with similar cars in the dataset
- Detailed price analysis and factors affecting the price

## Technical Stack
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Matplotlib
- Seaborn
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar navigation to access different features:
   - Data Overview: Explore the dataset and visualizations
   - Data Preprocessing: Configure and preview data preprocessing
   - Model Training & Evaluation: Train and compare different models
   - Predictions: Make price predictions for new cars

## Dataset
The project uses a dataset of used cars with the following features:
- Brand
- Model
- Model Year
- Mileage
- Engine specifications (horsepower, size, cylinders)
- Fuel type
- Transmission
- Accident history
- Clean title status
- Price

## Model Performance
The application allows you to compare the performance of different models:
- Linear Regression: Baseline model for price prediction
- Random Forest: Ensemble model with good accuracy
- XGBoost: Advanced gradient boosting model

## Acknowledgments
- Dataset source: [Add dataset source]
- Streamlit for the web application framework
- Scikit-learn and XGBoost for machine learning capabilities
- Plotly for interactive visualizations 