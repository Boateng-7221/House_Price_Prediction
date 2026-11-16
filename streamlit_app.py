"""
House Price Prediction Web App
Built with Streamlit for interactive predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Prediction App")
st.markdown("""
    Welcome to the House Price Prediction Application!
    
    This app uses a **Random Forest Machine Learning model** trained on house data
    to predict house prices based on key features. Enter your house details on the left
    and get an instant price prediction!
    """)

# Sidebar - Input section
st.sidebar.header("📝 Enter House Details")

# Create columns for better layout
col1, col2 = st.sidebar.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=1000,
        max_value=5000,
        value=2000,
        step=100,
        help="Enter the house area in square feet"
    )
    
    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,
        max_value=8,
        value=3,
        help="Number of bedrooms"
    )
    
    age = st.slider(
        "Age (years)",
        min_value=1,
        max_value=50,
        value=10,
        help="Age of the property in years"
    )

with col2:
    bathrooms = st.slider(
        "Bathrooms",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of bathrooms"
    )
    
    location = st.selectbox(
        "Location",
        ["Downtown", "Suburb", "Rural"],
        help="Select the property location"
    )
    
    parking = st.slider(
        "Parking Spaces",
        min_value=0,
        max_value=5,
        value=2,
        help="Number of parking spaces"
    )

# Location encoding
location_map = {"Downtown": 0, "Suburb": 1, "Rural": 2}
location_encoded = location_map[location]

# Load model
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    model_path = 'house_price_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("⚠️ Model file not found! Please train the model first.")
        return None

# Load metadata
@st.cache_resource
def load_metadata():
    """Load model metadata"""
    metadata_path = 'model_metadata.pkl'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    return None

# Load model and metadata
model = load_model()
metadata = load_metadata()

# Main prediction area
if model is not None:
    # Create input dataframe
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'location_encoded': [location_encoded],
        'age': [age],
        'parking': [parking]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction with fancy styling
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 💰 Predicted Price")
        st.metric(
            label="",
            value=f"${prediction:,.2f}",
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # Display input summary
    st.subheader("📊 Property Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Area", f"{area:,} sq ft")
        st.metric("Bedrooms", f"{bedrooms}")
        st.metric("Location", location)
    
    with summary_col2:
        st.metric("Bathrooms", f"{bathrooms}")
        st.metric("Age", f"{age} years")
        st.metric("Parking", f"{parking} spaces")
    
    with summary_col3:
        if metadata:
            st.metric("Model Type", metadata['model_type'])
            st.metric("R² Score", f"{metadata['r2_score']:.4f}")
            st.metric("Training Samples", metadata['training_samples'])
    
    # Price breakdown
    st.subheader("💡 How Features Impact Price")
    
    st.markdown("""
    **Most Important Features (from Random Forest analysis):**
    - 🎯 **Area** (37.3%) - Larger houses command higher prices
    - 🅿️ **Parking** (29.1%) - More parking increases value
    - 📅 **Age** (28.2%) - Newer properties are more expensive
    - 🛏️ **Bedrooms** (2.8%) - Minor impact on price
    - 🚿 **Bathrooms** (1.8%) - Minimal impact on price
    - 📍 **Location** (0.7%) - Slight impact on price
    """)

else:
    st.error("Model could not be loaded. Please ensure 'house_price_model.joblib' exists in the same directory.")

# Sidebar - Info section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About This App")
st.sidebar.info("""
    **Version:** 1.0
    
    **Model:** Random Forest Regressor
    
    **Features Used:**
    - Area (sq ft)
    - Bedrooms
    - Bathrooms
    - Location
    - Age
    - Parking
    
    **Performance:**
    - R² Score: 0.9715
    - RMSE: $25,682
    
    Built with ❤️ using Streamlit & Scikit-learn
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🎓 How It Works")
st.sidebar.markdown("""
    1. Enter your house details in the form
    2. The app uses a trained Random Forest model
    3. Prediction is based on patterns learned from 50 houses
    4. Accuracy: ~97% (R² = 0.9715)
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>🏠 House Price Prediction App | Built with Streamlit</p>
    <p><small>Disclaimer: Predictions are estimates based on historical data</small></p>
    </div>
    """, unsafe_allow_html=True)
