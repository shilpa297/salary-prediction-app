import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# XGBoost
from xgboost import XGBRegressor

# --------------------------------
# LOAD DATASET
# --------------------------------

print("Loading dataset...")

df = pd.read_csv("realistic_salary_dataset.csv")

print("Dataset Loaded Successfully!")

# --------------------------------
# KEEP IMPORTANT COLUMNS
# --------------------------------

df = df[[
    "Experience",
    "Qualifications",
    "Salary_INR",
    "Work Type",
    "Company Size",
    "Preference",
    "Job Title",
    "Role",
    "skills"
]]

print("\nUseful Columns Selected!")

# --------------------------------
# CREATE SEPARATE ENCODERS
# --------------------------------

qualification_encoder = LabelEncoder()

worktype_encoder = LabelEncoder()

preference_encoder = LabelEncoder()

job_encoder = LabelEncoder()

# --------------------------------
# APPLY ENCODING
# --------------------------------

df["Qualifications"] = qualification_encoder.fit_transform(
    df["Qualifications"]
)

df["Work Type"] = worktype_encoder.fit_transform(
    df["Work Type"]
)

df["Preference"] = preference_encoder.fit_transform(
    df["Preference"]
)

df["Job Title"] = job_encoder.fit_transform(
    df["Job Title"]
)

df["Role"] = job_encoder.transform(
    df["Role"]
)

print("\nCategorical Columns Encoded!")

# --------------------------------
# TF-IDF FOR SKILLS
# --------------------------------

print("\nApplying TF-IDF on skills...")

tfidf = TfidfVectorizer(
    max_features=100
)

skills_matrix = tfidf.fit_transform(df["skills"])

skills_df = pd.DataFrame(
    skills_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\nTF-IDF Completed!")

# --------------------------------
# COMBINE FEATURES
# --------------------------------

df.drop("skills", axis=1, inplace=True)

df = pd.concat(
    [
        df.reset_index(drop=True),
        skills_df.reset_index(drop=True)
    ],
    axis=1
)

print("\nFeatures Combined Successfully!")

# --------------------------------
# FEATURES AND TARGET
# --------------------------------

X = df.drop("Salary_INR", axis=1)

y = df["Salary_INR"]

print("\nFeatures and Target Prepared!")

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed!")

# --------------------------------
# XGBOOST MODEL
# --------------------------------

print("\nTraining XGBoost Model...")

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nXGBoost Model Trained Successfully!")

# --------------------------------
# PREDICTIONS
# --------------------------------

predictions = model.predict(X_test)

# --------------------------------
# MODEL EVALUATION
# --------------------------------

error = mean_absolute_error(
    y_test,
    predictions
)

# --------------------------------
# OUTPUT
# --------------------------------

print("\n✅ SAMPLE PREDICTIONS")
print(predictions[:10])

print("\n✅ MEAN ABSOLUTE ERROR (MAE)")
print(error)

# --------------------------------
# SAVE MODEL
# --------------------------------

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model Saved Successfully!")

# --------------------------------
# SAVE TF-IDF VECTORIZER
# --------------------------------

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("✅ Vectorizer Saved Successfully!")

# --------------------------------
# SAVE ENCODERS
# --------------------------------

with open("qualification_encoder.pkl", "wb") as file:
    pickle.dump(qualification_encoder, file)

with open("worktype_encoder.pkl", "wb") as file:
    pickle.dump(worktype_encoder, file)

with open("preference_encoder.pkl", "wb") as file:
    pickle.dump(preference_encoder, file)

with open("job_encoder.pkl", "wb") as file:
    pickle.dump(job_encoder, file)

print("✅ Encoders Saved Successfully!")