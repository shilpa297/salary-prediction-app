import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

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
# SALARY DISTRIBUTION
# --------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    df["Salary_INR"],
    bins=30,
    kde=True
)

plt.title("Salary Distribution")

plt.xlabel("Salary in INR")

plt.ylabel("Count")

plt.tight_layout()

plt.show()

# --------------------------------
# EXPERIENCE VS SALARY
# --------------------------------

plt.figure(figsize=(10, 5))

sns.scatterplot(
    x=df["Experience"],
    y=df["Salary_INR"]
)

plt.title("Experience vs Salary")

plt.xlabel("Experience")

plt.ylabel("Salary in INR")

plt.tight_layout()

plt.show()

# --------------------------------
# LABEL ENCODING
# --------------------------------

encoder = LabelEncoder()

df["Qualifications"] = encoder.fit_transform(df["Qualifications"])

df["Work Type"] = encoder.fit_transform(df["Work Type"])

df["Preference"] = encoder.fit_transform(df["Preference"])

# --------------------------------
# CORRELATION HEATMAP
# --------------------------------

numeric_df = df[[
    "Experience",
    "Qualifications",
    "Work Type",
    "Company Size",
    "Preference",
    "Salary_INR"
]]

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

# --------------------------------
# TOP PAYING JOB ROLES
# --------------------------------

top_roles = df.groupby("Role")["Salary_INR"].mean()

top_roles = top_roles.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_roles.values,
    y=top_roles.index
)

plt.title("Top Paying Job Roles")

plt.xlabel("Average Salary in INR")

plt.ylabel("Role")

plt.tight_layout()

plt.show()

# --------------------------------
# QUALIFICATION VS SALARY
# --------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df["Qualifications"],
    y=df["Salary_INR"]
)

plt.title("Qualification vs Salary")

plt.xlabel("Qualification")

plt.ylabel("Salary in INR")

plt.tight_layout()

plt.show()

print("\nEDA Completed Successfully!")