import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(layout="wide", page_title="Employee Attrition",page_icon="🎈")
st.title(":rainbow[Employee Attrition]🎈")
st.divider()

# Load models and preprocessing artifacts
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("random_forest_model.pkl")
    lr_model = joblib.load("logistic_regression_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return {"Random Forest": rf_model, "Logistic Regression": lr_model, "XGBoost": xgb_model}, scaler, feature_names

models, scaler, feature_names = load_artifacts()

# Align input features with the trained model
def align_features(input_df, feature_names):
    input_df = pd.get_dummies(input_df, drop_first=True)
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[feature_names]
    return input_df

# User input for manual predictions
def user_input_features():
    st.sidebar.header("Employee Details")
    features = {
        'Age': st.sidebar.slider("Age", 18, 65, 30),
        'DistanceFromHome': st.sidebar.slider("Distance From Home", 1, 30, 10),
        'MonthlyIncome': st.sidebar.slider("Monthly Income", 1000, 20000, 5000),
        'TotalWorkingYears': st.sidebar.slider("Total Working Years", 0, 40, 10),
        'YearsAtCompany': st.sidebar.slider("Years at Company", 0, 40, 5),
        'JobSatisfaction': st.sidebar.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4]),
        'EnvironmentSatisfaction': st.sidebar.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4]),
        'OverTime': st.sidebar.selectbox("OverTime", ['Yes', 'No']),
    }
    return pd.DataFrame([features])

# Main App

# Input Method
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    input_df = user_input_features()
    st.write("### Employee Details", input_df)
else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", input_df)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Align features and scale
aligned_input_df = align_features(input_df, feature_names)
scaled_input_df = scaler.transform(aligned_input_df)

# Model selection
st.sidebar.header("Select Model")
selected_model_name = st.sidebar.selectbox("", list(models.keys()))
selected_model = models[selected_model_name]

# Make predictions
try:
    predictions = selected_model.predict(scaled_input_df)
    probabilities = selected_model.predict_proba(scaled_input_df)[:, 1]
except AttributeError:
    st.error(f"The selected model '{selected_model_name}' does not support probability predictions.")
    st.stop()

# Add Yes/No labels for predictions
aligned_input_df['Attrition Prediction'] = ['Yes' if prob > 0.5 else 'No' for prob in probabilities]
aligned_input_df['Attrition Probability'] = probabilities

# Debugging: Ensure probabilities are valid
if not all(0 <= prob <= 1 for prob in probabilities):
    st.error("Error: Probabilities are not in the range [0, 1].")
    st.stop()

# Visualize Probabilities
st.write("### Attrition Probability Visualization")
fig = px.bar(
    aligned_input_df,
    x='Attrition Probability',
    y=aligned_input_df.index.astype(str),  # Ensure indices are treated as strings
    color='Attrition Prediction',  # Color bars by Yes/No
    orientation='h',  # Horizontal bar chart
    title="Attrition Probabilities with Results",
    labels={"y": "Employee Index", "x": "Attrition Probability", "color": "Attrition Result"},
    text='Attrition Probability'  # Show values on the bars
)

fig.update_layout(
    xaxis=dict(range=[0, 1]),
    yaxis_title="Employee Index",
    xaxis_title="Probability",
    title_x=0.5,  # Center the title
    height=600  # Adjust chart height for better readability
)

# Display chart
st.plotly_chart(fig)
