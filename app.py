# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Digital Twin Clinical Trial Simulator", layout="wide")

# st.title("Digital Twin Clinical Trial Simulator")
# st.subheader("Simulate clinical trials and virtual patients before real-world testing")
# st.sidebar.header("Simulation Settings")
# num_patients=st.sidebar.slider("Number of Patients", 50, 500, 100)
# trial_duration=st.sidebar.selectbox("Trail Duration (Months)",[3,6,12])
# treatment_arms=st.sidebar.multiselect("Treatment Arms",["Standard Drug","New Drug"], default=["Standard Drug", "New Drug"])
# run_simulation=st.sidebar.button("Run Simulation")
# st.sidebar.markdown("Developed by [Your Name](https://yourwebsite.com)")

# st.markdown("### Simulation Summary")
# col1, col2, col3 =st.columns(3)
# col1.metric("Patients", num_patients)
# col2.metric("Duration (Months)", trial_duration)
# col3.metric("Treatment Arms", len(treatment_arms))  

# if run_simulation:
#     st.success("Simulation completed successfully!")

#     months=np.arange(1, trial_duration+1)
#     for treatment in treatment_arms:
#         plt.plot(months, np.random.randint(80,150,size=trial_duration),label=treatment)
#     plt.xlabel("Months")
#     plt.ylabel("Blood Sugar Level")
#     plt.title("Simulated Blood Sugar Levels Over Time")
#     plt.legend()
#     st.pyplot(plt)  

#     data={
#         "Patient ID": [f"P{i+1}" for i in range(num_patients)],
#         "Treatment": np.random.choice(treatment_arms, num_patients),
#         "Outcome":np.random.choice(["Improved","No Change","Worsened"], num_patients)   

#     }
#     df =pd.DataFrame(data)
#     st.dataframe(df)

# st.markdown("---")
# st.markdown("Developed by **Your Name** | Powered by Python & Streamlit")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import os

# # -----------------------------
# # Page Config
# # -----------------------------
# st.set_page_config(page_title="Digital Twin Clinical Trial Simulator", layout="wide")

# st.title("🧪 Digital Twin Clinical Trial Simulator")
# st.subheader("Predict outcomes for real patients using a trained diabetes model")

# # -----------------------------
# # Load trained model
# # -----------------------------
# model_path = "diabetes_model.pkl"

# if not os.path.exists(model_path):
#     st.error(f"Model file not found: {model_path}. Please train the model first.")
# else:
#     model = joblib.load(model_path)

# # -----------------------------
# # Upload CSV of new patients
# # -----------------------------
# uploaded_file = st.file_uploader("Upload a CSV of patients", type=["csv"])

# if uploaded_file:
#     real_patients = pd.read_csv(uploaded_file)
#     st.write("Preview of Uploaded Patients:")
#     st.dataframe(real_patients.head())

#     # Check required columns
#     required_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
#                         'Insulin','BMI','DiabetesPedigreeFunction','Age']
#     if all(col in real_patients.columns for col in required_columns):
#         # Predict outcomes
#         predictions = model.predict(real_patients[required_columns])
#         real_patients["Predicted Outcome"] = ["Diabetic" if x==1 else "Non-Diabetic" for x in predictions]

#         st.subheader("Predicted Outcomes")
#         st.dataframe(real_patients)

#         # Plot outcomes
#         outcome_counts = real_patients['Predicted Outcome'].value_counts()
#         fig, ax = plt.subplots()
#         ax.bar(outcome_counts.index, outcome_counts.values, color=['green','red'])
#         ax.set_ylabel("Number of Patients")
#         ax.set_title("Predicted Diabetes Outcomes")
#         st.pyplot(fig)

#         # Download predictions
#         st.download_button(
#             label="Download Predictions as CSV",
#             data=real_patients.to_csv(index=False),
#             file_name="predicted_patients.csv",
#             mime="text/csv"
#         )
#     else:
#         st.error(f"Uploaded CSV must include these columns: {required_columns}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from faker import Faker

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Digital Twin Clinical Trial Simulator", layout="wide")
st.title("🧪 Digital Twin Clinical Trial Simulator")
fake = Faker()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#FFFFFF'>🏥 Digital Twin</h2>", unsafe_allow_html=True)
    st.button("Home")
    st.button("Patients")
    st.button("Charts")
    st.button("Settings")
disease = st.sidebar.radio("Select Disease", ["Diabetes", "Cancer"])
num_synthetic = st.sidebar.number_input("Number of synthetic patients to simulate", min_value=0, max_value=500, value=20)
simulate_treatment = st.sidebar.checkbox("Simulate Treatment Effects", value=True)
time_steps = st.sidebar.slider("Time steps for simulation", 1, 12, 3)

# -----------------------------
# Load Models
# -----------------------------
models = {}
cancer_features = None

if disease == "Diabetes":
    if os.path.exists("diabetes_model.pkl"):
        models['diabetes'] = joblib.load("diabetes_model.pkl")
    else:
        st.error("Diabetes model not found. Train first.")
elif disease == "Cancer":
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    rf = RandomForestClassifier()
    rf.fit(X, y)
    models['cancer'] = rf
    cancer_features = X.columns.tolist()

# -----------------------------
# Upload Real Patient Data
# -----------------------------
uploaded_file = st.file_uploader(f"Upload {disease} Patient CSV", type=["csv"])
real_patients = None
if uploaded_file:
    real_patients = pd.read_csv(uploaded_file)
    st.subheader("Real Patient Data Preview")
    st.dataframe(real_patients.head())

# -----------------------------
# Generate Synthetic Digital Twins
# -----------------------------
def generate_synthetic_diabetes(n):
    return pd.DataFrame({
        "Pregnancies": np.random.randint(0,10,n),
        "Glucose": np.random.randint(80,200,n),
        "BloodPressure": np.random.randint(50,100,n),
        "SkinThickness": np.random.randint(15,50,n),
        "Insulin": np.random.randint(15,276,n),
        "BMI": np.round(np.random.uniform(18,40,n),1),
        "DiabetesPedigreeFunction": np.round(np.random.uniform(0.1,2.5,n),2),
        "Age": np.random.randint(21,80,n)
    })

def generate_synthetic_cancer(n):
    if real_patients is not None and cancer_features is not None:
        return pd.DataFrame({
            col: np.random.uniform(real_patients[col].min(), real_patients[col].max(), n)
            for col in cancer_features
        })
    return None

synthetic_patients = None
if num_synthetic > 0:
    if disease == "Diabetes":
        synthetic_patients = generate_synthetic_diabetes(num_synthetic)
    elif disease == "Cancer":
        synthetic_patients = generate_synthetic_cancer(num_synthetic)

    if synthetic_patients is not None:
        synthetic_patients['Patient Name'] = [fake.name() for _ in range(len(synthetic_patients))]
        st.subheader(f"Synthetic Digital Twins ({num_synthetic} patients)")
        st.dataframe(synthetic_patients.head())

# -----------------------------
# Combine Real + Synthetic
# -----------------------------
all_patients = None
if real_patients is not None and synthetic_patients is not None:
    all_patients = pd.concat([real_patients, synthetic_patients], ignore_index=True)
elif real_patients is not None:
    all_patients = real_patients.copy()
elif synthetic_patients is not None:
    all_patients = synthetic_patients.copy()

# Add Patient Names if missing
if all_patients is not None and 'Patient Name' not in all_patients.columns:
    all_patients['Patient Name'] = [fake.name() for _ in range(len(all_patients))]

# -----------------------------
# Predict Outcomes
# -----------------------------
def predict(df, model, disease_type):
    if disease_type == "Diabetes":
        features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        df["Predicted Outcome"] = ["Diabetic" if x==1 else "Non-Diabetic" for x in model.predict(df[features])]
        df["Risk Score"] = model.predict_proba(df[features])[:,1]
    elif disease_type == "Cancer":
        df["Predicted Outcome"] = ["Malignant" if x==1 else "Benign" for x in model.predict(df[cancer_features])]
        df["Risk Score"] = model.predict_proba(df[cancer_features])[:,1]
    return df

def risk_category(score):
    if score >= 0.8:
        return "High Risk"
    elif score >= 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"

if all_patients is not None and disease.lower() in models:
    all_patients = predict(all_patients, models[disease.lower()], disease)
    all_patients['Risk Category'] = all_patients['Risk Score'].apply(risk_category)

# -----------------------------
# Treatment Simulation (Digital Twin)
# -----------------------------
if simulate_treatment and disease == "Diabetes" and all_patients is not None:
    for t in range(1, time_steps+1):
        all_patients[f'Glucose_t{t}'] = all_patients['Glucose'] * np.random.uniform(0.9,0.95)
        all_patients[f'BMI_t{t}'] = all_patients['BMI'] * np.random.uniform(0.95,1.0)

# -----------------------------
# Multi-chart Dashboard
# -----------------------------
if all_patients is not None:
    st.subheader("Cohort Overview")
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.dataframe(all_patients[['Patient Name','Predicted Outcome','Risk Score','Risk Category']].head(20))
    with col2:
        risk_counts = all_patients['Risk Category'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(risk_counts.index, risk_counts.values, color=['green','orange','red'])
        ax.set_ylabel("Number of Patients")
        ax.set_title("Risk Category Distribution")
        st.pyplot(fig)
    
    # -----------------------------
    # Tabs for Multiple Charts
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Outcome Distribution","Feature Correlation","Time Simulation"])
    
    with tab1:
        outcome_counts = all_patients['Predicted Outcome'].value_counts()
        st.bar_chart(outcome_counts)
    
    with tab2:
        if disease == "Diabetes":
            st.scatter_chart(all_patients[['BMI','Glucose']])
        elif disease == "Cancer":
            st.scatter_chart(all_patients[[cancer_features[0], cancer_features[1]]])
    
    with tab3:
        time_cols = [col for col in all_patients.columns if 'Glucose_t' in col]
        if time_cols:
            st.line_chart(all_patients[time_cols].head(10).T)

# -----------------------------
# Patient Detail Page
# -----------------------------
if all_patients is not None:
    st.subheader("Patient Detail")
    patient_names = all_patients['Patient Name'].tolist()
    selected_patient = st.selectbox("Select a patient to view details", patient_names)
    
    if selected_patient:
        patient_data = all_patients[all_patients['Patient Name']==selected_patient].iloc[0]
        st.write(f"### {selected_patient} Profile")
        st.write(f"**Age:** {patient_data.get('Age','N/A')}")
        st.write(f"**Predicted Outcome:** {patient_data['Predicted Outcome']}")
        st.write(f"**Risk Score:** {patient_data['Risk Score']:.2f} ({patient_data['Risk Category']})")
        
        st.write("#### Clinical Trial Eligibility")
        if patient_data['Risk Score'] > 0.5:
            st.success("Patient is eligible for intervention trial")
        else:
            st.info("Patient is at low risk; trial not recommended")
        
        # Show key features
        if disease == "Diabetes":
            features = ['Glucose','BMI','BloodPressure','Insulin']
        elif disease == "Cancer":
            features = cancer_features[:5]
        st.bar_chart(patient_data[features])
        
        # Show time-based simulation if exists
        time_cols = [col for col in all_patients.columns if col.endswith(tuple([str(i) for i in range(1,time_steps+1)]))]
        if time_cols:
            patient_time_data = patient_data[time_cols]
            st.line_chart(patient_time_data)
            
# -----------------------------
# Download
# -----------------------------
if all_patients is not None:
    st.download_button(
        label="Download Predictions as CSV",
        data=all_patients.to_csv(index=False),
        file_name=f"digital_twin_{disease.lower()}_predictions.csv",
        mime="text/csv"
    )
