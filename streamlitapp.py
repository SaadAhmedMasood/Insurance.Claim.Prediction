import streamlit as st
import numpy as np
import pickle

pipeline=pickle.load(open('pipeline.pkl','rb'))

# Define the Streamlit app
def main():    
    st.set_page_config(
    page_title="Insurance Claim Prediction",
    page_icon="⚕️",layout="wide"
    )
    st.title('Insurance Claim Prediction')
    
    with st.container():
        col1,col2= st.columns(spec=[0.5,0.5], gap="small")
    
    with col1:
        st.image("https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/in/wp-content/uploads/2022/11/health-insurance-image-scaled.jpg",use_column_width=True)
        st.write("Welcome to Insurance Claim Prediction App! This application is designed to assist you in estimating the expected insurance claim amount based on various factors. Whether you're an insurance agent, policyholder, or anyone interested in insurance analytics, this tool provides valuable insights into potential claim amounts")
        
        st.subheader("How It Works:")
        st.write("**Input Your Information:** Enter your age, BMI (Body Mass Index), blood pressure, number of children, gender, diabetic status, smoker status, and region.")
        st.write("**Get Your Prediction:** Advanced machine learning models analyze the provided information to predict the expected insurance claim amount.")
        
        st.subheader("Key Features:")
        st.write("**Predictive Models:** Employ sophisticated machine learning algorithms, including Linear Regression, Random Forest Regression, Gradient Boosting Regression, XGBoost Regression, and Support Vector Regression, to make accurate predictions.")
        st.write("**User-Friendly Interface:** The streamlined interface makes it easy to input your data and receive instant predictions.")
        st.write("**Comprehensive Insights:** Gain valuable insights into factors influencing insurance claim amounts and make informed decisions.")
        
        st.subheader("Benefit to use this APP:")
        st.write("**Efficiency:** Save time and resources by obtaining quick and accurate insurance claim predictions.")
        st.write("**Decision Support:** Whether you're an insurance professional assessing risk or an individual planning for future expenses,this app provides valuable guidance.")
        st.write("**Accessibility:** Accessible from any device with an internet connection, ensuring convenience wherever you are.")
        st.info("Disclaimer:Please note that the predictions provided by this application are based on statistical models and may not represent actual claim amounts. Always consult with a qualified insurance professional for personalized advice.")
        st.success("Created by:**Mr. Saad Ahmed Masood**")
        
    with col2:       
    # Collect user inputs
        age = st.slider('**Age**', min_value=18, max_value=120, value=30)
        bmi = st.number_input('**BMI**', min_value=10.0, max_value=50.0, value=25.0)
        blood_pressure = st.number_input('**Blood Pressure**', min_value=50, max_value=200, value=120)
        children = st.number_input('**Number of Children**', min_value=0, max_value=10, value=0)
        gender = st.radio('**Gender**', ['Male', 'Female'])
        diabetic = st.radio('**Diabetic**', ['Yes', 'No'])
        smoker = st.radio('**Smoker**', ['Yes', 'No'])
        region = st.selectbox('**Region**', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

    # Preprocess user inputs
        gender = 0 if gender == 'Male' else 1
        diabetic = 1 if diabetic == 'Yes' else 0
        smoker = 1 if smoker == 'Yes' else 0
        region_mapping = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}
        region = region_mapping[region]

    # Make prediction by deployment of our Model.
        if st.button('**Predict Claim Amount**'):
            prediction = predict_claim_amount(age, bmi, blood_pressure, children, gender, diabetic, smoker, region)
            st.success(f'Predicted Insurance Claim Amount: ${prediction:.2f}')        


def predict_claim_amount(age, bmi, blood_pressure, children, gender, diabetic, smoker, region):
    # Preprocess user inputs
    gender = 1 if gender == 'male' else 0
    diabetic = 1 if diabetic == 'Yes' else 0
    smoker = 1 if smoker == 'Yes' else 0
    region_mapping = {'northwest': 0, 'southeast': 1, 'southwest': 2}
    region_str = str(region)  # Convert to string to ensure it has the strip and lower methods
    region_str = region_str.strip().lower()  # Convert to lowercase and remove leading/trailing whitespace
    region = region_mapping.get(region_str, 0)  # Use get method to handle KeyError

    # Create feature array with one-hot encoding
    gender_male = 1 if gender == 1 else 0
    gender_female = 1 if gender == 0 else 0
    diabetic_yes = 1 if diabetic == 1 else 0
    diabetic_no = 1 if diabetic == 0 else 0
    smoker_yes = 1 if smoker == 1 else 0
    smoker_no = 1 if smoker == 0 else 0
    region_northwest = 1 if region == 0 else 0
    region_southeast = 1 if region == 1 else 0
    region_southwest = 1 if region == 2 else 0

    # Create feature array
    features = np.array([[age, bmi, blood_pressure, children, gender_male, gender_female, diabetic_yes, diabetic_no,
                          smoker_yes, smoker_no, region_northwest, region_southeast, region_southwest]])

    # Make prediction
    predicted_claim_amount = pipeline.predict(features)
    return predicted_claim_amount[0]

if __name__ == '__main__':
    main()