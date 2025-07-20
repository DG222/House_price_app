import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
df = pd.read_csv("train.csv")

# Select important features
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
categorical_features = ['MSZoning', 'Neighborhood', 'HouseStyle']
features = numeric_features + categorical_features
target = 'SalePrice'

# Prepare data
X = df[features]
y = df[target]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Build model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X, y)

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump((model_pipeline, features, categorical_features), f)

print("✅ Model trained and saved as house_model.pkl")