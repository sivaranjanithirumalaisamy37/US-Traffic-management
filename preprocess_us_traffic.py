import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------
# LOAD DATA
# -------------------
df = pd.read_csv("us_regional_traffic_dataset_5000.csv")

print("Original Shape:", df.shape)
print(df.head())

# -------------------
# CLEAN COLUMN NAMES
# -------------------
df.columns = df.columns.str.lower().str.strip()

# -------------------
# REMOVE DUPLICATES
# -------------------
df = df.drop_duplicates()

# -------------------
# HANDLE MISSING VALUES
# -------------------
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------
# OUTLIER CAPPING (IQR)
# -------------------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# -------------------
# ENCODE CATEGORICAL
# -------------------
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# -------------------
# SCALE NUMERIC
# -------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------
# SAVE CLEAN DATA
# -------------------
df.to_csv("us_traffic_cleaned.csv", index=False)

print("\n✅ Preprocessing Completed")
print("Saved file: us_traffic_cleaned.csv")
print(df.head())
