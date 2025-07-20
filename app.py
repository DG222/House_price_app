import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="House Price Estimator 🏡", layout="wide")

# Load model
with open("house_model.pkl", "rb") as f:
    model_pipeline, features, cat_features = pickle.load(f)

# Options for dropdowns
MSZoning_values = ['RL', 'RM', 'C (all)', 'FV', 'RH']
Neighborhood_values = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes',
                       'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR',
                       'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill',
                       'Blmngtn', 'BrDale', 'SWISU', 'Blueste']
HouseStyle_values = ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin']

st.title("🏡 House Price Estimator")
st.markdown("### Enter house details to estimate its sale price")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        GrLivArea = st.number_input("Living Area (sq ft)", value=1500)
        GarageCars = st.slider("Garage Capacity", 0, 5, 2)
        GarageArea = st.number_input("Garage Area (sq ft)", value=400)

    with col2:
        TotalBsmtSF = st.number_input("Basement Area (sq ft)", value=800)
        FullBath = st.slider("Number of Bathrooms", 0, 4, 2)
        YearBuilt = st.slider("Year Built", 1900, 2025, 2000)

    MSZoning = st.selectbox("Zoning", MSZoning_values)
    Neighborhood = st.selectbox("Neighborhood", Neighborhood_values)
    HouseStyle = st.selectbox("House Style", HouseStyle_values)

    submitted = st.form_submit_button("💰 Predict")

    if submitted:
        input_data = pd.DataFrame([{
            'OverallQual': OverallQual,
            'GrLivArea': GrLivArea,
            'GarageCars': GarageCars,
            'GarageArea': GarageArea,
            'TotalBsmtSF': TotalBsmtSF,
            'FullBath': FullBath,
            'YearBuilt': YearBuilt,
            'MSZoning': MSZoning,
            'Neighborhood': Neighborhood,
            'HouseStyle': HouseStyle
        }])

        prediction = model_pipeline.predict(input_data)[0]
        st.success(f"Estimated Sale Price: ₹ {prediction:,.0f}")