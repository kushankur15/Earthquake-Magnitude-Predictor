import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
import joblib   

filename = 'earthquake.csv'
df = pd.read_csv(filename)
df.dropna(subset=['mag'], inplace=True)
df['magError'].fillna(df['magError'].mean(), inplace=True)
df['magNst'].fillna(df['magNst'].mean(), inplace=True)
df['horizontalError'].fillna(df['horizontalError'].mean(), inplace=True)
df['nst'].fillna(df['nst'].mean(), inplace=True)
df['gap'].fillna(df['gap'].mean(), inplace=True)
df['dmin'].fillna(df['dmin'].mean(), inplace=True)
mag_source = df.groupby('magSource')['mag'].mean().reset_index()
# numeric cols
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('mag')
numeric_cols.remove('magError')
numeric_cols.remove('magNst')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()[1:-1]
categorical_cols.remove('magType')
categorical_cols.remove('id')
categorical_cols.remove('place')
categorical_cols.remove('updated')
categorical_cols.remove('type')
#scaling numeric cols
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
encoder = OneHotEncoder(sparse_output=False, drop=None)

encoded = encoder.fit_transform(df[categorical_cols])

# Create DataFrame with proper column names
encoded_df = pd.DataFrame(
    encoded, 
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

# Concatenate the original DataFrame including the encoded DataFrame
df = pd.concat([df, encoded_df], axis=1)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
input_cols = list(train_df.columns)[1:]
input_cols.remove('mag')
input_cols.remove('magError')
input_cols.remove('magNst')
input_cols.remove('magType')
input_cols.remove('id')
input_cols.remove('place')
input_cols.remove('updated')
input_cols.remove('type')
input_cols.remove('status')
input_cols.remove('magSource')
input_cols.remove('locationSource')
target_col = df['mag']
input_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Drop target columns (you don't want to predict using them)
for col in ['mag', 'magError', 'magNst']:
    if col in input_cols:
        input_cols.remove(col)
target_col = train_df['mag']
lr = LinearRegression()
lr.fit(train_df[input_cols], target_col)

new_eq = {
    "latitude": 38.322,
    "longitude": -118.433,
    "depth": 7.6,
    "nst": 45,
    "gap": 32.0,
    "dmin": 0.045,
    "rms": 0.76,
    "horizontalError": 1.2,
    "depthError": 0.8,
    "magError": 0.1,
    "magNst": 40,
    "net": "ci",
    "status": "reviewed",
    "locationSource": "ci"
}

def predict_magnitude(new_eq):
    new_eq_df = pd.DataFrame([new_eq])

    # Step 2: scale numeric features
    new_eq_df[numeric_cols] = scaler.transform(new_eq_df[numeric_cols])

    # Step 3: encode categoricals
    encoded_new = encoder.transform(new_eq_df[categorical_cols])
    encoded_new_df = pd.DataFrame(
        encoded_new,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=new_eq_df.index
    )

    # Step 4: combine ONLY numeric + encoded categoricals
    new_eq_final = pd.concat([new_eq_df[numeric_cols], encoded_new_df], axis=1)

    # 🚨 IMPORTANT: drop the raw categoricals so 'net', 'status', 'locationSource' never leak through
    # (this is why you got the error before)
    # new_eq_final should contain only numeric and encoded features now

    # Step 5: reorder columns to match training
    new_eq_final = new_eq_final.reindex(columns=input_cols, fill_value=0)
    # Step 6: predict
    predicted_mag = lr.predict(new_eq_final)
    return f"{predicted_mag[0]:.2f}"

print(predict_magnitude(new_eq))

# Save the trained model
joblib.dump(lr, "linear_regression_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Save the encoder
joblib.dump(encoder, "encoder.pkl")

# Also save the lists of columns used
joblib.dump(numeric_cols, "numeric_cols.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")
joblib.dump(input_cols, "input_cols.pkl")