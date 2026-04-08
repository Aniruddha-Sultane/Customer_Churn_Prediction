import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from model_trainer import ChurnModeler

# Page Configuration
st.set_page_config(page_title="Customer Churn Prediction System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stHeader {
        background: linear-gradient(135deg, #FF7E5F 0%, #FEB47B 100%);
        padding: 20px;
        border-radius: 12px;
        color: #000;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF7E5F 0%, #FEB47B 100%);
        color: #000;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Application Logo and Description
st.markdown('<div class="stHeader">📊 Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown("Analyze customer behavior and predict the likelihood of churn using advanced Machine Learning models.")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("👤 Customer Features")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 35)
    tenure = st.number_input("Tenure (Years)", 0, 10, 2)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 100.0, 5000.0, 1200.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    predict_btn = st.button("🚀 Predict Churn")

with col2:
    st.header("📈 Model Insights")
    
    # Static Metrics based on user's request
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown('<div class="metric-card">Random Forest<br/><strong>87% Acc</strong></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-card">Logistic Regression<br/><strong>82% Acc</strong></div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown('<div class="metric-card">Decision Tree<br/><strong>79% Acc</strong></div>', unsafe_allow_html=True)
    
    st.divider()

    # Model Performance Visualization
    performance_data = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'Decision Tree'],
        'Accuracy': [0.87, 0.82, 0.79]
    })
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Accuracy', y='Model', data=performance_data, palette='flare', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title('Model Performance Comparison')
    st.pyplot(fig)

    # Output Prediction Result
    if predict_btn:
        with st.spinner("Analyzing customer data..."):
            try:
                # Load models (In a real app, you would pre-train and save them)
                # For demonstration, we'll assume they are loaded via joblib.
                # If they don't exist, we'll run the training script.
                try:
                    scaler = joblib.load('scaler.joblib')
                    best_model = joblib.load('churn_model.joblib')
                    le = joblib.load('label_encoder.joblib')
                except:
                    # Run generation and trainer scripts first if they don't exist
                    # (Simplified for this sandbox environment)
                    st.warning("Models not found. Training models first...")
                    import os
                    os.system("python generate_data.py")
                    os.system("python model_trainer.py")
                    scaler = joblib.load('scaler.joblib')
                    best_model = joblib.load('churn_model.joblib')
                    le = joblib.load('label_encoder.joblib')

                # Preprocess user input
                gender_encoded = 1 if gender == "Female" else 0  # Simplified
                contract_encoded = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2) # Simplified
                
                input_data = np.array([[gender_encoded, age, tenure, monthly_charges, total_charges, contract_encoded]])
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = best_model.predict(input_scaled)[0]
                prediction_prob = best_model.predict_proba(input_scaled)[0][1]

                st.divider()
                if prediction == 1:
                    st.error(f"⚠️ High Risk of Churn! ({prediction_prob:.2%} probability)")
                else:
                    st.success(f"✅ Customer is likely to STAY! (Churn risk: {prediction_prob:.2%})")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Sidebar info
st.sidebar.divider()
st.sidebar.info("Developed with Scikit-learn for churn prediction analysis.")