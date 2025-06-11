import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import re

# Set page configuration
st.set_page_config(page_title="Used Cars Price Prediction", layout="wide")

# Add title and description
st.title("Used Cars Price Prediction")
st.write("""
This app analyzes used car data and predicts prices using different machine learning models.
Explore the data, compare model performance, and visualize predictions.
""")

# Load the dataset
@st.cache_data
def load_data():
    # Load the raw dataset
    df = pd.read_csv('used_cars.csv')
    
    # Clean price column - extract numerical values
    df['price_clean'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('"', '', regex=False).astype(float)
    
    # Clean mileage column - extract numerical values
    df['mileage_clean'] = df['milage'].str.replace(',', '', regex=False).str.extract(r'(\d+)').astype(float)
    
    # Extract engine information
    df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').fillna(0).astype(float)
    df['engine_size'] = df['engine'].str.extract(r'(\d+\.?\d*)L').fillna(0).astype(float)
    df['cylinders'] = df['engine'].str.extract(r'(\d+) Cylinder').fillna(0).astype(float)
    
    # Create binary features for accident and clean title
    df['has_accident'] = df['accident'].str.contains('accident', case=False, na=False).astype(int)
    df['has_clean_title'] = df['clean_title'].str.contains('Yes', case=False, na=False).astype(int)
    
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Preprocessing", "Model Training & Evaluation", "Predictions"])

# Data Overview page
if page == "Data Overview":
    st.header("Data Overview")
    
    # Display basic information
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of records:** {df.shape[0]}")
        st.write(f"**Number of features:** {df.shape[1]}")
    with col2:
        st.write(f"**Missing values:** {df.isnull().sum().sum()}")
        st.write(f"**Duplicate records:** {df.duplicated().sum()}")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    # Data visualizations
    st.subheader("Data Visualizations")
    
    # Price distribution
    st.write("**Price Distribution**")
    fig = px.histogram(df, x="price_clean", nbins=50, title="Price Distribution")
    fig.update_layout(xaxis_title="Price ($)", yaxis_title="Count")
    st.plotly_chart(fig)
    
    # Brand distribution
    st.write("**Top 15 Car Brands**")
    brand_counts = df['brand'].value_counts().head(15)
    fig = px.bar(x=brand_counts.index, y=brand_counts.values, title="Top 15 Car Brands")
    fig.update_layout(xaxis_title="Brand", yaxis_title="Count")
    st.plotly_chart(fig)
    
    # Price by brand
    st.write("**Average Price by Brand (Top 15)**")
    brand_avg_price = df.groupby('brand')['price_clean'].mean().sort_values(ascending=False).head(15)
    fig = px.bar(x=brand_avg_price.index, y=brand_avg_price.values, title="Average Price by Brand")
    fig.update_layout(xaxis_title="Brand", yaxis_title="Average Price ($)")
    st.plotly_chart(fig)
    
    # Price vs. mileage scatter plot
    st.write("**Price vs. Mileage**")
    fig = px.scatter(df, x="mileage_clean", y="price_clean", opacity=0.6, title="Price vs. Mileage")
    fig.update_layout(xaxis_title="Mileage", yaxis_title="Price ($)")
    st.plotly_chart(fig)
    
    # Price vs. model year
    st.write("**Price vs. Model Year**")
    year_avg_price = df.groupby('model_year')['price_clean'].mean().reset_index()
    fig = px.line(year_avg_price, x="model_year", y="price_clean", title="Average Price by Model Year")
    fig.update_layout(xaxis_title="Model Year", yaxis_title="Average Price ($)")
    st.plotly_chart(fig)

# Data Preprocessing page
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    # Display preprocessing steps
    st.subheader("Preprocessing Steps")
    st.write("""
    1. **Price Cleaning**: Extracted numerical values from price strings
    2. **Mileage Cleaning**: Extracted numerical values from mileage strings
    3. **Engine Information Extraction**:
       - Extracted horsepower
       - Extracted engine size (in liters)
       - Extracted number of cylinders
    4. **Categorical Feature Encoding**:
       - Created binary features for accidents and clean title
    5. **Missing Value Handling**:
       - Numerical features: Imputed with median
       - Categorical features: Imputed with most frequent value
    """)
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    # Feature selection
    st.write("**Select Features for Modeling**")
    
    # Identify numerical and categorical columns
    numerical_cols = ['model_year', 'mileage_clean', 'horsepower', 'engine_size', 'cylinders', 'has_accident', 'has_clean_title']
    categorical_cols = ['brand', 'fuel_type', 'transmission']
    
    # Feature selection options
    selected_num_features = st.multiselect(
        "Select Numerical Features",
        options=numerical_cols,
        default=numerical_cols
    )
    
    selected_cat_features = st.multiselect(
        "Select Categorical Features",
        options=categorical_cols,
        default=categorical_cols
    )
    
    # Display selected features
    st.write(f"**Total Selected Features:** {len(selected_num_features) + len(selected_cat_features)}")
    
    # Preprocessing preview
    if st.button("Preview Preprocessed Data"):
        st.subheader("Preprocessed Data Preview")
        
        # Create a copy of the dataframe with selected features
        selected_features = selected_num_features + selected_cat_features + ['price_clean']
        df_selected = df[selected_features].copy()
        
        # Display info about missing values
        st.write("**Missing Values Before Imputation**")
        missing_values = df_selected.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            st.dataframe(missing_values)
        else:
            st.write("No missing values found in selected features.")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, selected_num_features),
                ('cat', categorical_transformer, selected_cat_features)
            ])
        
        # Fit the preprocessor
        X = df_selected.drop('price_clean', axis=1)
        y = df_selected['price_clean']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the preprocessor on training data
        preprocessor.fit(X_train)
        
        # Display feature names after preprocessing
        st.write("**Features After Preprocessing**")
        
        # Get feature names from one-hot encoding
        cat_features = []
        if selected_cat_features:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = ohe.get_feature_names_out(selected_cat_features).tolist()
        
        all_features = selected_num_features + cat_features
        st.write(f"Total features after preprocessing: {len(all_features)}")
        
        # Display correlation with price
        st.subheader("Correlation with Price (Numerical Features)")
        corr = df_selected[selected_num_features + ['price_clean']].corr()['price_clean'].sort_values(ascending=False)
        fig = px.bar(x=corr.index, y=corr.values, title="Correlation with Price")
        fig.update_layout(xaxis_title="Feature", yaxis_title="Correlation Coefficient")
        st.plotly_chart(fig)

# Model Training & Evaluation page
elif page == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    # Model selection
    st.subheader("Select Models")
    models_to_train = st.multiselect(
        "Choose models to train",
        options=["Linear Regression", "Random Forest", "XGBoost"],
        default=["Linear Regression", "Random Forest", "XGBoost"]
    )
    
    # Feature selection
    numerical_cols = ['model_year', 'mileage_clean', 'horsepower', 'engine_size', 'cylinders', 'has_accident', 'has_clean_title']
    categorical_cols = ['brand', 'fuel_type', 'transmission']
    
    # Feature selection options
    selected_num_features = st.multiselect(
        "Select Numerical Features",
        options=numerical_cols,
        default=['model_year', 'mileage_clean', 'horsepower']
    )
    
    selected_cat_features = st.multiselect(
        "Select Categorical Features",
        options=categorical_cols,
        default=['brand', 'fuel_type', 'transmission']
    )
    
    # Train models button
    if st.button("Train Models"):
        # Create a copy of the dataframe with selected features
        selected_features = selected_num_features + selected_cat_features + ['price_clean']
        df_selected = df[selected_features].copy()
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, selected_num_features),
                ('cat', categorical_transformer, selected_cat_features)
            ])
        
        # Prepare data
        X = df_selected.drop('price_clean', axis=1)
        y = df_selected['price_clean']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        models = {}
        if "Linear Regression" in models_to_train:
            models["Linear Regression"] = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('regressor', LinearRegression())])
        
        if "Random Forest" in models_to_train:
            models["Random Forest"] = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        
        if "XGBoost" in models_to_train:
            models["XGBoost"] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])
        
        # Train and evaluate models
        results = {}
        predictions = {}
        
        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Store predictions
                predictions[name] = {
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Store results
                results[name] = {
                    'Train RMSE': train_rmse,
                    'Test RMSE': test_rmse,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae
                }
        
        # Display results
        st.subheader("Model Performance")
        
        # Create a dataframe for results
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)
        
        # Plot metrics comparison
        st.subheader("Metrics Comparison")
        
        # RMSE comparison
        fig = go.Figure(data=[
            go.Bar(name='Train RMSE', x=list(results.keys()), y=[results[model]['Train RMSE'] for model in results]),
            go.Bar(name='Test RMSE', x=list(results.keys()), y=[results[model]['Test RMSE'] for model in results])
        ])
        fig.update_layout(title="RMSE Comparison", xaxis_title="Model", yaxis_title="RMSE")
        st.plotly_chart(fig)
        
        # R² comparison
        fig = go.Figure(data=[
            go.Bar(name='Train R²', x=list(results.keys()), y=[results[model]['Train R²'] for model in results]),
            go.Bar(name='Test R²', x=list(results.keys()), y=[results[model]['Test R²'] for model in results])
        ])
        fig.update_layout(title="R² Comparison", xaxis_title="Model", yaxis_title="R²")
        st.plotly_chart(fig)
        
        # Actual vs Predicted plots
        st.subheader("Actual vs Predicted")
        
        for name in results.keys():
            fig = px.scatter(x=predictions[name]['y_test'], y=predictions[name]['y_pred'],
                           title=f"{name}: Actual vs Predicted")
            fig.update_layout(xaxis_title="Actual Price", yaxis_title="Predicted Price")
            
            # Add perfect prediction line
            x_range = [predictions[name]['y_test'].min(), predictions[name]['y_test'].max()]
            fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='Perfect Prediction',
                                   line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig)
        
        # Feature importance for tree-based models
        st.subheader("Feature Importance")
        
        if "Random Forest" in models:
            rf_model = models["Random Forest"]
            preprocessor = rf_model.named_steps['preprocessor']
            rf_regressor = rf_model.named_steps['regressor']
            
            # Get feature names after preprocessing
            cat_features = []
            if selected_cat_features:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = ohe.get_feature_names_out(selected_cat_features).tolist()
            
            all_features = selected_num_features + cat_features
            
            # Get feature importances
            importances = rf_regressor.feature_importances_
            
            # Create a dataframe for feature importances
            if len(all_features) == len(importances):
                feature_importance = pd.DataFrame({
                    'Feature': all_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importances
                fig = px.bar(feature_importance.head(15), x='Feature', y='Importance',
                           title="Random Forest: Top 15 Feature Importances")
                st.plotly_chart(fig)
            else:
                st.write("Feature names and importances length mismatch. Cannot display feature importances.")
        
        if "XGBoost" in models:
            xgb_model = models["XGBoost"]
            preprocessor = xgb_model.named_steps['preprocessor']
            xgb_regressor = xgb_model.named_steps['regressor']
            
            # Get feature names after preprocessing
            cat_features = []
            if selected_cat_features:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = ohe.get_feature_names_out(selected_cat_features).tolist()
            
            all_features = selected_num_features + cat_features
            
            # Get feature importances
            importances = xgb_regressor.feature_importances_
            
            # Create a dataframe for feature importances
            if len(all_features) == len(importances):
                feature_importance = pd.DataFrame({
                    'Feature': all_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importances
                fig = px.bar(feature_importance.head(15), x='Feature', y='Importance',
                           title="XGBoost: Top 15 Feature Importances")
                st.plotly_chart(fig)
            else:
                st.write("Feature names and importances length mismatch. Cannot display feature importances.")

# Predictions page
elif page == "Predictions":
    st.header("Make Predictions")
    
    # Create input form for user to enter car details
    st.subheader("Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", options=sorted(df['brand'].unique()))
        model_year = st.number_input("Model Year", min_value=1990, max_value=2023, value=2018)
        mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000)
        horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=300)
        engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
    
    with col2:
        fuel_type = st.selectbox("Fuel Type", options=sorted([x for x in df['fuel_type'].unique() if isinstance(x, str)]))
        transmission = st.selectbox("Transmission", options=sorted(df['transmission'].unique()))
        has_accident = st.selectbox("Accident History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        has_clean_title = st.selectbox("Clean Title", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Model selection for prediction
    model_for_prediction = st.selectbox(
        "Select Model for Prediction",
        options=["Linear Regression", "Random Forest", "XGBoost"],
        index=1  # Default to Random Forest
    )
    
    # Make prediction button
    if st.button("Predict Price"):
        # Create a dataframe with the input values
        input_data = pd.DataFrame({
            'brand': [brand],
            'model_year': [model_year],
            'mileage_clean': [mileage],
            'horsepower': [horsepower],
            'engine_size': [engine_size],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'has_accident': [has_accident],
            'has_clean_title': [has_clean_title]
        })
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_cols = ['model_year', 'mileage_clean', 'horsepower', 'engine_size', 'has_accident', 'has_clean_title']
        categorical_cols = ['brand', 'fuel_type', 'transmission']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Prepare full dataset for training
        X = df[numerical_cols + categorical_cols]
        y = df['price_clean']
        
        # Train the selected model
        if model_for_prediction == "Linear Regression":
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', LinearRegression())])
        elif model_for_prediction == "Random Forest":
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        else:  # XGBoost
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])
        
        # Train the model
        model.fit(X, y)
        
        # Make prediction
        prediction = model.predict(input_data[numerical_cols + categorical_cols])[0]
        
        # Display prediction
        st.success(f"### Predicted Price: ${prediction:,.2f}")
        
        # Display prediction explanation
        st.subheader("Prediction Explanation")
        
        # Create a comparison with similar cars
        st.write("**Similar Cars in Dataset**")
        
        # Filter similar cars
        similar_cars = df[
            (df['brand'] == brand) & 
            (df['model_year'] >= model_year - 3) & 
            (df['model_year'] <= model_year + 3)
        ].sort_values('mileage_clean').head(5)
        
        if not similar_cars.empty:
            # Display similar cars
            similar_cars_display = similar_cars[['brand', 'model', 'model_year', 'milage', 'price']].copy()
            st.dataframe(similar_cars_display)
            
            # Calculate average price of similar cars
            avg_price = similar_cars['price_clean'].mean()
            st.write(f"**Average price of similar cars:** ${avg_price:,.2f}")
            
            # Compare prediction with average
            diff = prediction - avg_price
            diff_percent = (diff / avg_price) * 100
            
            if abs(diff_percent) < 10:
                st.write("Your predicted price is close to the average price of similar cars.")
            elif diff > 0:
                st.write(f"Your predicted price is {diff_percent:.1f}% higher than the average price of similar cars.")
            else:
                st.write(f"Your predicted price is {abs(diff_percent):.1f}% lower than the average price of similar cars.")
                
            # Add a comparison graph
            st.write("**Price Comparison Graph**")
            
            # Create data for the graph
            comparison_data = pd.DataFrame({
                'Category': ['Your Car', 'Average Similar Cars'],
                'Price': [prediction, avg_price]
            })
            
            # Create a bar chart for price comparison
            fig = px.bar(
                comparison_data,
                x='Category',
                y='Price',
                title='Predicted Price vs. Average Similar Cars',
                color='Category',
                color_discrete_sequence=['#1E88E5', '#FFC107'],
                text_auto=True
            )
            
            # Format the y-axis to show dollar amounts
            fig.update_layout(
                yaxis_title="Price ($)",
                xaxis_title="",
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                height=400
            )
            
            # Add a horizontal line for the predicted price
            fig.add_hline(
                y=prediction,
                line_dash="dash",
                line_color="#1E88E5",
                annotation_text="Your Car's Predicted Price",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Remove the "Mileage vs. Price for Similar Cars" section
            # The code for this section has been removed
        else:
            st.write("No similar cars found in the dataset.")
        
        # Display factors affecting the price
        st.write("**Factors Affecting the Price**")
        
        # Create a list of factors
        factors = []
        
        # Model year
        avg_year = df['model_year'].mean()
        if model_year > avg_year:
            factors.append(f"✅ Newer model year (+{model_year - avg_year:.0f} years above average)")
        else:
            factors.append(f"❌ Older model year ({model_year - avg_year:.0f} years below average)")
        
        # Mileage
        avg_mileage = df['mileage_clean'].mean()
        if mileage < avg_mileage:
            factors.append(f"✅ Lower mileage ({(avg_mileage - mileage):,.0f} miles below average)")
        else:
            factors.append(f"❌ Higher mileage ({(mileage - avg_mileage):,.0f} miles above average)")
        
        # Accident history
        if has_accident == 0:
            factors.append("✅ No accident history")
        else:
            factors.append("❌ Has accident history")
        
        # Clean title
        if has_clean_title == 1:
            factors.append("✅ Clean title")
        else:
            factors.append("❌ No clean title")
        
        # Display factors
        for factor in factors:
            st.write(factor)