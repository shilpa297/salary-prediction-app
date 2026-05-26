import streamlit as st
import pandas as pd
import pickle

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="AI Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------------
# LOAD TF-IDF VECTORIZER
# --------------------------------

with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

# --------------------------------
# LOAD ENCODERS
# --------------------------------

with open("qualification_encoder.pkl", "rb") as file:
    qualification_encoder = pickle.load(file)

with open("worktype_encoder.pkl", "rb") as file:
    worktype_encoder = pickle.load(file)

with open("preference_encoder.pkl", "rb") as file:
    preference_encoder = pickle.load(file)

with open("job_encoder.pkl", "rb") as file:
    job_encoder = pickle.load(file)

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("💡 About Project")

st.sidebar.info(
    """
    This AI Salary Prediction App uses:

    ✅ XGBoost Regression  
    ✅ TF-IDF NLP  
    ✅ Streamlit  
    ✅ Machine Learning  
    """
)

# --------------------------------
# TITLE
# --------------------------------

st.title("💼 AI Salary Prediction App")

st.write(
    "Enter job details and predict estimated salary using AI."
)

# --------------------------------
# CREATE 2 COLUMNS
# --------------------------------

col1, col2 = st.columns(2)

# --------------------------------
# LEFT COLUMN
# --------------------------------

with col1:

    experience = st.slider(
        "Years of Experience",
        0,
        15,
        3
    )

    qualification = st.selectbox(
        "Qualification",
        list(qualification_encoder.classes_)
    )

    work_type = st.selectbox(
        "Work Type",
        list(worktype_encoder.classes_)
    )

    company_size = st.number_input(
        "Company Size",
        min_value=50,
        max_value=200000,
        value=1000
    )

# --------------------------------
# RIGHT COLUMN
# --------------------------------

with col2:

    preference = st.selectbox(
        "Preference",
        list(preference_encoder.classes_)
    )

    job_title = st.selectbox(
        "Job Role",
        list(job_encoder.classes_)
    )

    skills = st.text_area(
        "Enter Skills (comma separated)",
        "Python, AWS, Docker"
    )

# --------------------------------
# ENCODE USER INPUTS
# --------------------------------

qualification_encoded = qualification_encoder.transform(
    [qualification]
)[0]

work_type_encoded = worktype_encoder.transform(
    [work_type]
)[0]

preference_encoded = preference_encoder.transform(
    [preference]
)[0]

job_encoded = job_encoder.transform(
    [job_title]
)[0]

# --------------------------------
# TF-IDF TRANSFORM
# --------------------------------

skills_vector = tfidf.transform([skills])

skills_df = pd.DataFrame(
    skills_vector.toarray(),
    columns=tfidf.get_feature_names_out()
)

# --------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------

input_data = pd.DataFrame({
    "Experience": [experience],
    "Qualifications": [qualification_encoded],
    "Work Type": [work_type_encoded],
    "Company Size": [company_size],
    "Preference": [preference_encoded],
    "Job Title": [job_encoded],
    "Role": [job_encoded]
})

# --------------------------------
# COMBINE ALL FEATURES
# --------------------------------

final_input = pd.concat(
    [
        input_data.reset_index(drop=True),
        skills_df.reset_index(drop=True)
    ],
    axis=1
)

# --------------------------------
# PREDICTION BUTTON
# --------------------------------

if st.button("🚀 Predict Salary"):

    prediction = model.predict(final_input)

    salary = int(prediction[0])

    st.success(
        f"💰 Predicted Salary: ₹{salary:,}"
    )

    st.metric(
        label="Estimated Annual Salary",
        value=f"₹{salary:,}"
    )

# --------------------------------
# MODEL COMPARISON
# --------------------------------

st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        201023,
        182870,
        176723
    ]
})

st.dataframe(comparison_df)

# --------------------------------
# FOOTER
# --------------------------------

st.write("---")

st.write(
    "Built using XGBoost + NLP + Streamlit"
)