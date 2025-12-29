import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
rf = joblib.load("rf_model.pkl")

# Header
st.markdown('<div class="main-header">💰 Medical Insurance Charges Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🏥 Advanced ML Model to Estimate Your Insurance Costs</div>', unsafe_allow_html=True)

# Info section
with st.expander("ℹ️ How It Works", expanded=False):
    st.info("""
    This application uses a **Random Forest Machine Learning model** to predict medical insurance charges 
    based on patient characteristics. Simply enter your details below and get an instant prediction!
    """)

# Main content in columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📋 Patient Information")
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    age = st.slider("👤 Age", 18, 100, 40, help="Your current age")
    
    gender = st.selectbox("👨‍👩 Gender", ["Male", "Female"], help="Select your gender")
    is_female = 1 if gender == "Female" else 0
    
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", 10.0, 60.0, 30.0, step=0.1, help="Your BMI value")
    
    children = st.selectbox("👶 Number of Children", [0, 1, 2, 3, 4, 5], help="Dependent children")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🏥 Health & Location")
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    smoker = st.selectbox("🚭 Smoking Status", ["No", "Yes"], help="Do you smoke?")
    is_smoker = 1 if smoker == "Yes" else 0
    
    region = st.selectbox("🗺️ Region", ["Southeast", "Other"], help="Your region")
    region_southeast = 1 if region == "Southeast" else 0
    
    bmi_category_Obese = 1 if bmi >= 30 else 0
    
    # Display BMI category
    if bmi < 18.5:
        bmi_status = "🟢 Underweight"
    elif bmi < 25:
        bmi_status = "🟢 Normal"
    elif bmi < 30:
        bmi_status = "🟡 Overweight"
    else:
        bmi_status = "🔴 Obese"
    
    st.markdown(f'<div class="info-box">BMI Status: <strong>{bmi_status}</strong></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Separator
st.divider()

# Prediction section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🎯 Get Prediction", key="predict_btn", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age],
            'is_female': [is_female],
            'bmi': [bmi],
            'children': [children],
            'is_smoker': [is_smoker],
            'region_southeast': [region_southeast],
            'bmi_category_Obese': [bmi_category_Obese]
        })

        prediction = rf.predict(input_data)[0]
        
        # Display prediction with custom styling
        st.markdown(f'<div class="prediction-box">💵 ${prediction:,.2f}</div>', unsafe_allow_html=True)
        
        # Show additional insights
        st.markdown("---")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.metric("Annual Cost", f"${prediction:,.0f}")
        
        with insight_col2:
            st.metric("Monthly Cost", f"${prediction/12:,.0f}")
        
        with insight_col3:
            st.metric("Smoking Impact", f"{'High ⚠️' if is_smoker else 'None ✅'}")
        
        # Risk assessment
        if prediction > 50000:
            st.warning("⚠️ Your predicted charges are significantly high. Consider lifestyle changes.")
        elif prediction > 30000:
            st.info("ℹ️ Your predicted charges are moderate. Health insurance is important!")
        else:
            st.success("✅ Your predicted charges are relatively low.")
