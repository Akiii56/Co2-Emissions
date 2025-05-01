# Importing libraries-----------------------------------------------------------------------------------------
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="CO2 Emissions Prediction",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Creating Sidebar-------------------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=100)
    st.markdown("# CO2 Emissions by Vehicle")
    st.markdown("---")
    user_input = st.selectbox('Select Analysis Type', ('Visualization', 'Model'))
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application helps predict CO2 emissions based on vehicle specifications.
    - Select 'Visualization' to explore the dataset
    - Select 'Model' to predict CO2 emissions
    """)

# Load the vehicle dataset
df = pd.read_csv('co2 Emissions.csv')

# Preprocess the data (Handle missing values and outliers)
df['Fuel Consumption Comb (L/100 km)'] = df['Fuel Consumption Comb (L/100 km)'].fillna(df['Fuel Consumption Comb (L/100 km)'].mean())

# Remove rows with natural gas as fuel type
fuel_type_mapping = {"Z": "Premium Gasoline", "X": "Regular Gasoline", "D": "Diesel", "E": "Ethanol(E85)", "N": "Natural Gas"}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)

# Apply Logarithmic Transformation
df_natural['Engine Size(L)'] = np.log(df_natural['Engine Size(L)'] + 1)
df_natural['Fuel Consumption Comb (L/100 km)'] = np.log(df_natural['Fuel Consumption Comb (L/100 km)'] + 1)
df_natural['CO2 Emissions(g/km)'] = np.log(df_natural['CO2 Emissions(g/km)'] + 1)

# Remove Outliers using Z-Score method
df_cleaned = df_natural[(np.abs(stats.zscore(df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']])) < 3).all(axis=1)]

# Visualization Section-------------------------------------------------------------------------------------------------
if user_input == 'Visualization':
    st.title('🌍 CO2 Emissions Analysis Dashboard')
    
    # Dataset Overview
    st.header("📊 Dataset Overview")
    with st.expander("View Dataset", expanded=False):
        st.dataframe(df_cleaned.style.background_gradient(cmap='YlOrRd'))

    # Brand Analysis
    st.header("🚗 Car Brands Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        df_brand = df_cleaned['Make'].value_counts().reset_index().rename(columns={'count':'Count'})
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_brand, x="Make", y="Count", palette="viridis")
        plt.xticks(rotation=75)
        plt.title("Distribution of Cars by Manufacturer")
        plt.xlabel("Manufacturers")
        plt.ylabel("Number of Cars")
        st.pyplot(fig)

    with col2:
        st.markdown("### Top Manufacturers")
        st.dataframe(df_brand.head(10).style.background_gradient(cmap='YlOrRd'))

    # Additional Visualizations
    st.header("📈 CO2 Emissions Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_cleaned, x='Engine Size(L)', y='CO2 Emissions(g/km)', hue='Fuel Type', palette='viridis')
        plt.title("Engine Size vs CO2 Emissions")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_cleaned, x='Cylinders', y='CO2 Emissions(g/km)', palette='viridis')
        plt.title("CO2 Emissions by Number of Cylinders")
        st.pyplot(fig)

else:
    # Model Section
    st.title('🌍 CO2 Emission Prediction Model')
    
    # Prepare the data for modeling
    X = df_cleaned[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_cleaned['CO2 Emissions(g/km)']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42).fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display metrics in a nice format
    st.header("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R-squared Score", f"{r2:.2%}")
    with col2:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    with col3:
        st.metric("Root Mean Squared Error", f"{rmse:.4f}")

    # Prediction Interface
    st.header("🚗 Predict CO2 Emissions")
    st.markdown("Enter the vehicle specifications below to predict CO2 emissions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        engine_size = st.number_input('Engine Size (L)', min_value=0.1, max_value=10.0, step=0.1, format="%.1f")
        cylinders = st.number_input('Number of Cylinders', min_value=2, max_value=16, step=1)
    
    with col2:
        fuel_consumption = st.number_input('Fuel Consumption (L/100 km)', min_value=1.0, max_value=30.0, step=0.1, format="%.1f")
        
        if st.button('Predict CO2 Emissions', key='predict'):
            input_data = [[np.log(engine_size + 1), cylinders, np.log(fuel_consumption + 1)]]
            predicted_co2 = model.predict(input_data)
            
            st.markdown("---")
            st.markdown(f"### 🎯 Prediction Result")
            st.markdown(f"<div style='text-align: center; font-size: 24px; color: #1f77b4;'>"
                       f"Predicted CO2 Emissions: {np.exp(predicted_co2[0]):.2f} g/km</div>",
                       unsafe_allow_html=True)

    # Feature Importance
    st.header("📊 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance in CO2 Emission Prediction")
    st.pyplot(fig)

# Sample data snippet for testing the UI functionality
sample_data = {
    "Make": ["Toyota", "Ford", "BMW", "Honda", "Chevrolet", "Hyundai", "Nissan"],
    "Engine Size(L)": [2.0, 3.5, 3.0, 1.8, 2.4, 2.0, 2.5],
    "Cylinders": [4, 6, 6, 4, 4, 4, 4],
    "Fuel Consumption Comb (L/100 km)": [7.5, 10.5, 9.2, 6.8, 8.0, 7.2, 7.9],
    "Fuel Type": ["Regular Gasoline", "Diesel", "Premium Gasoline", "Regular Gasoline", "Ethanol(E85)", "Regular Gasoline", "Regular Gasoline"],
    "CO2 Emissions(g/km)": [180, 250, 220, 160, 200, 175, 190]
}
df_sample = pd.DataFrame(sample_data)
