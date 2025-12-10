"""
Real Estate Investment Advisor - Streamlit Application
======================================================

A user-friendly web application for predicting:
1. Classification: Is this a Good Investment? (Yes/No)
2. Regression: Estimated Property Price after 5 Years

Features:
- Interactive property input form
- Real-time predictions with confidence scores
- Feature importance visualization
- Property filtering and comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add src to path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import preprocessing pipeline
try:
    from preprocessing_pipeline import (
        apply_log_transformation,
        apply_winsorization,
        apply_frequency_encoding,
        apply_ordinal_encoding,
        apply_onehot_encoding,
        FEATURE_METADATA,
        ONEHOT_COLS,
        ORDINAL_COLS
    )
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    st.warning("Preprocessing module not found. Using basic preprocessing.")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .good-investment {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .bad-investment {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        color: #0a0a0a;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load the original dataset for reference values."""
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'india_housing_prices.csv'))
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Load classifier
    classifier_path = os.path.join(model_dir, 'gradient_boosting_model.pkl')
    if os.path.exists(classifier_path):
        try:
            models['classifier'] = joblib.load(classifier_path)
            st.success("✓ Classification model loaded")
        except Exception as e:
            st.error(f"Error loading classifier: {e}")
    
    # Load regressor (if exists)
    regressor_path = os.path.join(model_dir, 'best_regressor_model.pkl')
    if os.path.exists(regressor_path):
        try:
            models['regressor'] = joblib.load(regressor_path)
            st.success("✓ Regression model loaded")
        except Exception as e:
            st.error(f"Error loading regressor: {e}")
    
    # Load frequency maps
    freq_path = os.path.join(model_dir, 'freq_maps.pkl')
    if os.path.exists(freq_path):
        try:
            models['freq_maps'] = joblib.load(freq_path)
        except:
            models['freq_maps'] = None
    else:
        models['freq_maps'] = None
    
    return models

@st.cache_data
def get_unique_values(df, column):
    """Get unique values for a column."""
    if df is not None and column in df.columns:
        return sorted(df[column].unique().tolist())
    return []

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def calculate_derived_features(input_dict):
    """Calculate derived features from raw input."""
    # Price per SqFt
    if input_dict['Size_in_SqFt'] > 0:
        input_dict['Price_per_SqFt'] = (input_dict['Price_in_Lakhs'] * 100000) / input_dict['Size_in_SqFt']
    else:
        input_dict['Price_per_SqFt'] = 0
    
    # Age of Property
    current_year = 2025
    input_dict['Age_of_Property'] = current_year - input_dict['Year_Built']
    
    # School Category
    schools = input_dict['Nearby_Schools']
    if schools == 0:
        input_dict['School_Category'] = 'No Schools'
    elif schools <= 2:
        input_dict['School_Category'] = '1-2 Schools'
    elif schools <= 4:
        input_dict['School_Category'] = '3-4 Schools'
    elif schools <= 6:
        input_dict['School_Category'] = '5-6 Schools'
    else:
        input_dict['School_Category'] = '7+ Schools'
    
    # Hospital Category
    hospitals = input_dict['Nearby_Hospitals']
    if hospitals == 0:
        input_dict['Hospital_Category'] = 'No Hospitals'
    elif hospitals <= 2:
        input_dict['Hospital_Category'] = '1-2 Hospitals'
    elif hospitals <= 4:
        input_dict['Hospital_Category'] = '3-4 Hospitals'
    elif hospitals <= 6:
        input_dict['Hospital_Category'] = '5-6 Hospitals'
    else:
        input_dict['Hospital_Category'] = '7+ Hospitals'
    
    # Combined Amenities Category
    combined = input_dict['Nearby_Schools'] + input_dict['Nearby_Hospitals']
    if combined <= 2:
        input_dict['Combined_Category'] = 'Low (0-2)'
    elif combined <= 5:
        input_dict['Combined_Category'] = 'Medium (3-5)'
    elif combined <= 8:
        input_dict['Combined_Category'] = 'High (6-8)'
    else:
        input_dict['Combined_Category'] = 'Very High (9+)'
    
    # Combined Amenities numeric
    input_dict['Combined_Amenities'] = combined
    
    # Parking Space Numeric (Yes/No -> 1/0)
    parking = input_dict.get('Parking_Space', 'No')
    input_dict['Parking_Space_Numeric'] = 1 if parking == 'Yes' else 0
    
    # Price_per_SqFt transformations
    input_dict['Price_per_SqFt_log'] = np.log1p(input_dict['Price_per_SqFt'])
    input_dict['Price_per_SqFt_winsor'] = input_dict['Price_per_SqFt']  # Simplified for single sample
    
    return input_dict

def preprocess_for_model(input_df, models, df_reference):
    """
    Preprocess input data for model prediction.
    
    The model expects:
    - Categorical (OneHot): City, Locality, Amenities, School_Category, Hospital_Category, Combined_Category
    - Numerical (Scale): BHK, Size_in_SqFt, Price_in_Lakhs, Price_per_SqFt, Year_Built, Furnished_Status, 
                         Floor_No, Total_Floors, Age_of_Property, Nearby_Schools, Nearby_Hospitals, 
                         Public_Transport_Accessibility, Combined_Amenities, Parking_Space_Numeric, 
                         Price_per_SqFt_log, Price_per_SqFt_winsor, City_freq, Locality_freq
    - Plus one-hot encoded columns for: State, Property_Type, Parking_Space, Security, Facing, Owner_Type, Availability_Status
    
    Args:
        input_df: DataFrame with raw input
        models: Dictionary containing models and preprocessing artifacts
        df_reference: Reference DataFrame for frequency encoding
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df = input_df.copy()
    
    # Step 1: Frequency Encoding for City and Locality
    if df_reference is not None:
        for col in ['City', 'Locality']:
            freq_map = df_reference[col].value_counts(normalize=True).to_dict()
            df[col + '_freq'] = df[col].map(freq_map).fillna(1e-6)
    else:
        df['City_freq'] = 1e-6
        df['Locality_freq'] = 1e-6
    
    # Step 2: Ordinal Encoding for Furnished_Status and Public_Transport_Accessibility
    furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
    transport_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    df['Furnished_Status'] = df['Furnished_Status'].map(furnished_map).fillna(0)
    df['Public_Transport_Accessibility'] = df['Public_Transport_Accessibility'].map(transport_map).fillna(0)
    
    # Step 3: One-Hot Encoding for categorical columns (with drop_first=True)
    onehot_cols = ['State', 'Property_Type', 'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']
    
    # Get all possible categories from reference data
    if df_reference is not None:
        # Create one-hot columns matching the training data
        for col in onehot_cols:
            if col in df.columns:
                unique_vals = df_reference[col].unique()
                for val in unique_vals[1:]:  # Skip first for drop_first=True
                    col_name = f"{col}_{val}"
                    df[col_name] = (df[col] == val).astype(int)
    
    # Drop original categorical columns that were one-hot encoded
    for col in onehot_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Ensure Parking_Space_Numeric is present
    if 'Parking_Space_Numeric' not in df.columns:
        df['Parking_Space_Numeric'] = 0
    
    return df

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_investment(input_df, models, df_reference):
    """
    Make investment classification prediction.
    
    Returns:
        tuple: (prediction, probability, feature_importance)
    """
    if 'classifier' not in models:
        return None, None, None
    
    try:
        model = models['classifier']
        
        # Get feature names expected by model
        # The model pipeline handles preprocessing internally
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Get feature importance
        try:
            clf = model.named_steps.get('clf')
            if hasattr(clf, 'feature_importances_'):
                feature_importance = clf.feature_importances_
            else:
                feature_importance = None
        except:
            feature_importance = None
        
        return prediction, probability, feature_importance
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def predict_future_price(input_df, models):
    """
    Predict property price after 5 years using regression model.
    
    If regression model is not available, use simple appreciation formula.
    
    Returns:
        tuple: (predicted_price, confidence_interval)
    """
    if 'regressor' in models:
        try:
            model = models['regressor']
            prediction = model.predict(input_df)[0]
            return prediction, None
        except Exception as e:
            st.warning(f"Regression model error: {e}. Using formula-based estimate.")
    
    # Fallback: Use compound interest formula
    current_price = input_df['Price_in_Lakhs'].values[0]
    appreciation_rate = 0.08  # 8% annual appreciation
    years = 5
    future_price = current_price * ((1 + appreciation_rate) ** years)
    
    return future_price, (future_price * 0.9, future_price * 1.1)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar(df):
    """Render the sidebar with property filters."""
    st.sidebar.markdown("## 🔍 Property Filters")
    
    filters = {}
    
    # State filter
    if df is not None:
        states = ['All'] + get_unique_values(df, 'State')
        filters['state'] = st.sidebar.selectbox("State", states)
        
        # City filter (dependent on state)
        if filters['state'] != 'All':
            cities = ['All'] + get_unique_values(df[df['State'] == filters['state']], 'City')
        else:
            cities = ['All'] + get_unique_values(df, 'City')
        filters['city'] = st.sidebar.selectbox("City", cities)
        
        # Price range
        st.sidebar.markdown("### 💰 Price Range (Lakhs)")
        price_min = float(df['Price_in_Lakhs'].min())
        price_max = float(df['Price_in_Lakhs'].max())
        filters['price_range'] = st.sidebar.slider(
            "Select Range",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, min(price_max, 500.0))
        )
        
        # BHK filter
        bhk_options = ['All'] + sorted(df['BHK'].unique().tolist())
        filters['bhk'] = st.sidebar.selectbox("BHK", bhk_options)
        
        # Property Type filter
        property_types = ['All'] + get_unique_values(df, 'Property_Type')
        filters['property_type'] = st.sidebar.selectbox("Property Type", property_types)
    
    return filters

def render_input_form(df):
    """Render the property input form."""
    st.markdown("### 📝 Enter Property Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Location Details**")
        state = st.selectbox("State", get_unique_values(df, 'State') if df is not None else ['Maharashtra'])
        
        # Filter cities by state
        if df is not None:
            cities = get_unique_values(df[df['State'] == state], 'City')
        else:
            cities = ['Mumbai']
        city = st.selectbox("City", cities)
        
        # Filter localities by city
        if df is not None:
            localities = get_unique_values(df[df['City'] == city], 'Locality')
        else:
            localities = ['Andheri West']
        locality = st.selectbox("Locality", localities)
        
        property_type = st.selectbox("Property Type", 
            get_unique_values(df, 'Property_Type') if df is not None else 
            ['Apartment', 'Independent House', 'Villa'])
    
    with col2:
        st.markdown("**Property Specifications**")
        bhk = st.number_input("BHK", min_value=1, max_value=6, value=2, step=1)
        size = st.number_input("Size (Sq Ft)", min_value=100, max_value=10000, value=1000, step=50)
        price = st.number_input("Price (Lakhs)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
        year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2020, step=1)
        floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=2, step=1)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=60, value=10, step=1)
    
    with col3:
        st.markdown("**Amenities & Features**")
        furnished = st.selectbox("Furnished Status", ['Unfurnished', 'Semi-furnished', 'Furnished'])
        transport = st.selectbox("Public Transport Access", ['Low', 'Medium', 'High'])
        parking = st.selectbox("Parking", ['No', 'Yes'])
        security = st.selectbox("Security", ['No', 'Yes'])
        facing = st.selectbox("Facing", ['East', 'North', 'South', 'West'])
        owner = st.selectbox("Owner Type", ['Broker', 'Builder', 'Owner'])
        availability = st.selectbox("Availability", ['Ready_to_Move', 'Under_Construction'])
    
    st.markdown("**Neighborhood**")
    col4, col5 = st.columns(2)
    with col4:
        nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=3, step=1)
    with col5:
        nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=2, step=1)
    
    amenities = st.multiselect("Amenities", 
        ['Gym', 'Pool', 'Clubhouse', 'Garden', 'Playground', 'Power Backup', 'Lift', 'Security'],
        default=['Gym', 'Lift'])
    
    # Create input dictionary
    input_dict = {
        'State': state,
        'City': city,
        'Locality': locality,
        'Property_Type': property_type,
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Price_in_Lakhs': price,
        'Year_Built': year_built,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Furnished_Status': furnished,
        'Public_Transport_Accessibility': transport,
        'Parking_Space': parking,
        'Security': security,
        'Facing': facing,
        'Owner_Type': owner,
        'Availability_Status': availability,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Amenities': ', '.join(amenities) if amenities else 'None'
    }
    
    # Calculate derived features
    input_dict = calculate_derived_features(input_dict)
    
    return input_dict

def render_predictions(input_dict, models, df_reference):
    """Render prediction results."""
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_dict])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Classification: Investment Quality")
        
        if 'classifier' in models:
            try:
                # Prepare features for the model
                # The model was trained with specific preprocessing
                
                features_df = input_df.copy()
                
                # Ensure required columns are present
                # The model's preprocessor expects these categorical columns (OneHot):
                # City, Locality, Amenities, School_Category, Hospital_Category, Combined_Category
                
                # And these numerical columns (Scale):
                # BHK, Size_in_SqFt, Price_in_Lakhs, Price_per_SqFt, Year_Built, Furnished_Status,
                # Floor_No, Total_Floors, Age_of_Property, Nearby_Schools, Nearby_Hospitals,
                # Public_Transport_Accessibility, Combined_Amenities, Parking_Space_Numeric,
                # Price_per_SqFt_log, Price_per_SqFt_winsor, City_freq, Locality_freq
                
                # Plus one-hot encoded columns for State, Property_Type, Parking_Space, Security, 
                # Facing, Owner_Type, Availability_Status
                
                # Step 1: Frequency Encoding for City and Locality
                if df_reference is not None:
                    for col in ['City', 'Locality']:
                        freq_map = df_reference[col].value_counts(normalize=True).to_dict()
                        features_df[col + '_freq'] = features_df[col].map(freq_map).fillna(1e-6)
                else:
                    features_df['City_freq'] = 1e-6
                    features_df['Locality_freq'] = 1e-6
                
                # Step 2: Ordinal Encoding for Furnished_Status and Public_Transport_Accessibility
                furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
                transport_map = {'Low': 0, 'Medium': 1, 'High': 2}
                
                features_df['Furnished_Status'] = features_df['Furnished_Status'].map(furnished_map).fillna(0)
                features_df['Public_Transport_Accessibility'] = features_df['Public_Transport_Accessibility'].map(transport_map).fillna(0)
                
                # Step 3: One-Hot Encoding for categorical columns (matching training data)
                onehot_cols = ['State', 'Property_Type', 'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']
                
                # Get all possible categories from reference data and create one-hot columns
                if df_reference is not None:
                    for col in onehot_cols:
                        if col in features_df.columns and col in df_reference.columns:
                            unique_vals = sorted(df_reference[col].unique())
                            for val in unique_vals[1:]:  # Skip first for drop_first=True
                                col_name = f"{col}_{val}"
                                features_df[col_name] = (features_df[col] == val).astype(int)
                
                # Drop original categorical columns that were one-hot encoded
                for col in onehot_cols:
                    if col in features_df.columns:
                        features_df = features_df.drop(columns=[col])
                
                # Ensure all required columns exist with proper data types
                # The model pipeline will handle the actual encoding, but we need correct columns
                
                # Fill any missing columns with 0
                features_df = features_df.fillna(0)
                
                # Debug info (can be removed later)
                with st.expander("Debug: Input Features", expanded=False):
                    st.write("**Feature columns:**", features_df.columns.tolist())
                    st.write("**Shape:**", features_df.shape)
                
                # Make prediction using the pipeline
                model = models['classifier']
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0]
                
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box good-investment">
                        <h2 style="color: #4CAF50; margin: 0;">✅ GOOD INVESTMENT</h2>
                        <p style="margin: 0.5rem 0 0 0;">This property shows promising investment potential.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-box bad-investment">
                        <h2 style="color: #F44336; margin: 0;">⚠️ RISKY INVESTMENT</h2>
                        <p style="margin: 0.5rem 0 0 0;">This property may not be an ideal investment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence gauge
                confidence = max(probability) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={'text': "Confidence Score", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4CAF50" if prediction == 1 else "#F44336"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FFEBEE"},
                            {'range': [50, 75], 'color': "#FFF3E0"},
                            {'range': [75, 100], 'color': "#E8F5E9"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 2},
                            'thickness': 0.75,
                            'value': confidence
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Probability breakdown
                st.markdown("**Probability Breakdown:**")
                prob_df = pd.DataFrame({
                    'Category': ['Bad Investment', 'Good Investment'],
                    'Probability': probability * 100
                })
                fig_prob = px.bar(prob_df, x='Category', y='Probability', 
                                  color='Category', 
                                  color_discrete_map={'Good Investment': '#4CAF50', 'Bad Investment': '#F44336'})
                fig_prob.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
                
            except Exception as e:
                st.error(f"Classification error: {e}")
                st.info("Please ensure the model was trained with matching features.")
        else:
            st.warning("Classification model not loaded. Please ensure 'best_classifier_model.pkl' exists in the models folder.")
    
    with col2:
        st.markdown("### 📈 Regression: Future Price Estimate")
        
        # Calculate future price using formula
        current_price = input_dict['Price_in_Lakhs']
        appreciation_rate = 0.08  # 8% annual appreciation
        years = 5
        future_price = current_price * ((1 + appreciation_rate) ** years)
        
        # Display future price
        st.metric(
            label="Estimated Price in 5 Years",
            value=f"₹{future_price:.2f} Lakhs",
            delta=f"+₹{future_price - current_price:.2f} Lakhs ({((future_price/current_price)-1)*100:.1f}%)"
        )
        
        # Price projection chart
        years_range = list(range(0, 11))
        prices = [current_price * ((1 + appreciation_rate) ** y) for y in years_range]
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=years_range, 
            y=prices,
            mode='lines+markers',
            name='Projected Price',
            line=dict(color='#1E88E5', width=2),
            marker=dict(size=8)
        ))
        
        # Mark current and 5-year prediction
        fig_price.add_trace(go.Scatter(
            x=[0, 5],
            y=[current_price, future_price],
            mode='markers',
            name='Key Points',
            marker=dict(size=15, color=['#4CAF50', '#FF9800'], symbol='star')
        ))
        
        fig_price.update_layout(
            title="10-Year Price Projection",
            xaxis_title="Years from Now",
            yaxis_title="Price (Lakhs)",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Appreciation breakdown
        st.markdown("**Appreciation Summary:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Current Price", f"₹{current_price:.1f}L")
        with col_b:
            st.metric("5-Year Price", f"₹{future_price:.1f}L")
        with col_c:
            roi = ((future_price / current_price) - 1) * 100
            st.metric("Expected ROI", f"{roi:.1f}%")

def render_property_comparison(df, filters):
    """Render property comparison section."""
    st.markdown("---")
    st.markdown("## 📊 Market Comparison")
    
    if df is None:
        st.warning("No data available for comparison.")
        return
    
    # Apply filters
    filtered_df = df.copy()
    
    if filters.get('state') and filters['state'] != 'All':
        filtered_df = filtered_df[filtered_df['State'] == filters['state']]
    
    if filters.get('city') and filters['city'] != 'All':
        filtered_df = filtered_df[filtered_df['City'] == filters['city']]
    
    if filters.get('bhk') and filters['bhk'] != 'All':
        filtered_df = filtered_df[filtered_df['BHK'] == filters['bhk']]
    
    if filters.get('property_type') and filters['property_type'] != 'All':
        filtered_df = filtered_df[filtered_df['Property_Type'] == filters['property_type']]
    
    if filters.get('price_range'):
        filtered_df = filtered_df[
            (filtered_df['Price_in_Lakhs'] >= filters['price_range'][0]) &
            (filtered_df['Price_in_Lakhs'] <= filters['price_range'][1])
        ]
    
    if len(filtered_df) == 0:
        st.info("No properties match the selected filters.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Price Distribution")
        fig = px.histogram(filtered_df, x='Price_in_Lakhs', nbins=30,
                          title="Property Price Distribution",
                          color_discrete_sequence=['#1E88E5'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Price per SqFt by Property Type")
        fig = px.box(filtered_df, x='Property_Type', y='Price_per_SqFt',
                    title="Price per SqFt Distribution",
                    color='Property_Type')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("### Market Summary")
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("Avg Price", f"₹{filtered_df['Price_in_Lakhs'].mean():.1f}L")
    with col4:
        st.metric("Median Price", f"₹{filtered_df['Price_in_Lakhs'].median():.1f}L")
    with col5:
        st.metric("Avg Price/SqFt", f"₹{filtered_df['Price_per_SqFt'].mean():.0f}")
    with col6:
        st.metric("Properties", f"{len(filtered_df):,}")

def render_feature_importance(models):
    """Render feature importance visualization."""
    if 'classifier' not in models:
        return
    
    st.markdown("---")
    st.markdown("## 🔑 Feature Importance")
    
    try:
        model = models['classifier']
        clf = model.named_steps.get('clf')
        
        if hasattr(clf, 'feature_importances_'):
            # Get feature names from preprocessor
            preprocessor = model.named_steps.get('preprocess')
            
            feature_names = []
            
            # Get categorical feature names from OneHotEncoder
            if hasattr(preprocessor, 'named_transformers_'):
                if 'onehot' in preprocessor.named_transformers_:
                    encoder = preprocessor.named_transformers_['onehot']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_features = encoder.get_feature_names_out()
                        feature_names.extend(cat_features)
                
                # Get numerical feature names
                if 'scale' in preprocessor.named_transformers_:
                    # These are the numerical columns
                    num_features = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
                    feature_names.extend(num_features)
            
            importance = clf.feature_importances_
            
            # If we have feature names, use them
            if len(feature_names) == len(importance):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True).tail(15)
            else:
                # Use generic names
                importance_df = pd.DataFrame({
                    'Feature': [f'Feature {i}' for i in range(len(importance))],
                    'Importance': importance
                }).sort_values('Importance', ascending=True).tail(15)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Top 15 Most Important Features",
                        color='Importance',
                        color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.info(f"Feature importance visualization not available: {e}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<p class="main-header">🏠 Real Estate Investment Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicting Property Profitability & Future Value</p>', unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    models = load_models()
    
    # Sidebar filters
    filters = render_sidebar(df)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Investment Predictor", "📊 Market Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Analyze Your Property Investment")
        st.markdown("Enter property details below to get an investment recommendation and price forecast.")
        
        # Input form
        input_dict = render_input_form(df)
        
        # Predict button
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            predict_button = st.button("🔮 Analyze Investment", use_container_width=True, type="primary")
        
        if predict_button:
            with st.spinner("Analyzing property..."):
                render_predictions(input_dict, models, df)
                render_feature_importance(models)
    
    with tab2:
        st.markdown("### Market Overview")
        render_property_comparison(df, filters)
        
        # Additional market insights
        if df is not None:
            st.markdown("---")
            st.markdown("### 🏙️ Top Cities by Average Price")
            
            city_stats = df.groupby('City').agg({
                'Price_in_Lakhs': 'mean',
                'Price_per_SqFt': 'mean',
                'ID': 'count'
            }).rename(columns={'ID': 'Count'}).sort_values('Price_in_Lakhs', ascending=False).head(10)
            
            fig = px.bar(city_stats.reset_index(), x='City', y='Price_in_Lakhs',
                        title="Top 10 Cities by Average Property Price",
                        color='Price_in_Lakhs',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        ### About This Application
        
        **Real Estate Investment Advisor** is an AI-powered tool designed to help investors
        make data-driven decisions in the Indian real estate market.
        
        #### Features:
        
        - **🎯 Investment Classification**: Predicts whether a property is a good investment
          based on multiple factors including price, location, amenities, and market trends.
        
        - **📈 Price Forecasting**: Estimates property value after 5 years using historical
          appreciation rates and market analysis.
        
        - **📊 Market Analysis**: Provides comprehensive market insights with interactive
          visualizations and filtering options.
        
        #### How It Works:
        
        1. **Data Collection**: The model was trained on 250,000+ property records across India.
        
        2. **Feature Engineering**: 
           - Log transformation for price normalization
           - Frequency encoding for high-cardinality features (City, Locality)
           - Ordinal encoding for ordered categories
           - One-hot encoding for categorical variables
        
        3. **Machine Learning Models**:
           - **Classification**: Random Forest Classifier for investment quality prediction
           - **Regression**: Price prediction using compound appreciation model
        
        #### Investment Score Criteria:
        
        A property is classified as a **Good Investment** if:
        - Price is below median market price, OR
        - Price per SqFt is below median, OR
        - Investment Score ≥ 2 (based on BHK, furnishing, availability)
        
        ---
        
        **Disclaimer**: This tool provides estimates based on historical data and should not
        be considered as financial advice. Always consult with real estate professionals
        before making investment decisions.
        
        ---
        
        **Built with** ❤️ using Python, Streamlit, Scikit-learn, and Plotly
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>Real Estate Investment Advisor | © 2025 | Powered by Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
