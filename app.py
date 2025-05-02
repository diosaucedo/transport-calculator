# app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import base64

# === Constants ===
TOTAL_FIXED_COST = 13_044_792
TOTAL_ANNUAL_WEIGHT = 17_562_606
FIXED_COST_PER_LB = TOTAL_FIXED_COST / TOTAL_ANNUAL_WEIGHT
TRANSPORT_RATE = 0.02
FSD_GREEN = "#6BA539"
FSD_ORANGE = "#F7941D"

# === Load Logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_path = f"data:image/png;base64,{encoded_image}"

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_excel("Final_Quarterly.xlsx")
    df['Total_Weight'] = df['Estimated_Donated_Weight'] + df['Estimated_Purchased_Weight']
    df['Weight_Tier'] = 'Unassigned'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] < 8000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] >= 8000) & (df['Total_Weight'] <= 20000), 'Weight_Tier'] = 'Mid'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] > 20000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'BP') & (df['Total_Weight'] < 7311), 'Weight_Tier'] = 'All'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] < 6000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] >= 6000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] < 2000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] >= 2000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'PP'), 'Weight_Tier'] = 'All'
    return df[df['Weight_Tier'] != 'Unassigned']

# === Train Model ===
@st.cache_resource
def train_model(df):
    model_dict = {}
    scores = []
    features = ['Region', 'PROGRAM', 'Quarter', 'hh', 'Estimated_Donated_Weight', 'Estimated_Purchased_Weight']
    target = 'Cost'
    cat_cols = ['Region', 'PROGRAM', 'Quarter']
    num_cols = [col for col in features if col not in cat_cols]

    for (program, tier), subset in df.groupby(['PROGRAM', 'Weight_Tier']):
        if len(subset) < 20:
            continue
        X = subset[features]
        y = np.log1p(subset[target])
        mask = (~y.isna()) & (~np.isinf(y))
        X = X[mask]
        y = y[mask]

        pre = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ])
        pipe = Pipeline([
            ('pre', pre),
            ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200))
        ])
        pipe.fit(X, y)
        y_pred = np.expm1(pipe.predict(X))
        y_true = np.expm1(y)
        model_dict[(program, tier)] = pipe
        scores.append({
            'Program': program,
            'Weight_Tier': tier,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        })
    return model_dict, pd.DataFrame(scores)

# === Predict ===
def predict(model_dict, scores, program, region, hh, weight, donated_pct, miles):
    q_weight = weight / 4
    q_donated = q_weight * (donated_pct / 100.0)
    q_purchased = q_weight * (1 - (donated_pct / 100.0))

    if program == 'AGENCY':
        tier = 'Low' if q_weight < 8000 else 'Mid' if q_weight <= 20000 else 'High'
    elif program == 'MP':
        tier = 'Low' if q_weight < 6000 else 'High'
    elif program == 'SP':
        tier = 'Low' if q_weight < 2000 else 'High'
    elif program in ['BP', 'PP']:
        tier = 'All'
    else:
        return None

    model = model_dict.get((program, tier))
    if not model:
        return None

    row = scores[(scores['Program'] == program) & (scores['Weight_Tier'] == tier)]
    if row.empty:
        return None

    mae = row['MAE'].values[0]
    rmse = row['RMSE'].values[0]

    df_input = pd.DataFrame([{
        'Region': region,
        'PROGRAM': program,
        'Quarter': 'Q1',
        'hh': hh,
        'Estimated_Donated_Weight': q_donated,
        'Estimated_Purchased_Weight': q_purchased
    }])

    pred = np.expm1(model.predict(df_input)[0])
    base_cost = pred * 4
    fixed = weight * FIXED_COST_PER_LB
    transport = weight * miles * TRANSPORT_RATE
    total = base_cost + fixed + transport

    return {
        'Quarterly Cost': round(pred, 2),
        'Base Annual Cost': round(base_cost, 2),
        'Fixed Cost': round(fixed, 2),
        'Transport Cost': round(transport, 2),
        'Total Cost': round(total, 2),
        'Cost per lb': round(total / weight, 4),
        'Upper Bound MAE': round(total + mae * 4, 2),
        'Upper Bound RMSE': round(total + rmse * 4, 2)
    }

# === UI ===
st.set_page_config(page_title="Cost Estimator", layout="centered")
st.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{encoded_image}" style="height:110px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Cost Estimator</h2>", unsafe_allow_html=True)

region_options = {
    'North Coastal (27.7 mi)': 27.7,
    'North Inland (46.6 mi)': 46.6,
    'Central (19.2 mi)': 19.2,
    'East (60.5 mi)': 60.5,
    'South (28.3 mi)': 28.3
}

with st.form("estimator"):
    weight = st.number_input("1. Enter total weight (lbs):", min_value=1.0, value=2000.0)
    region_label = st.selectbox("2. Select delivery region:", list(region_options.keys()))
    program = st.selectbox("3. Select agency/program:", ["AGENCY", "SP", "BP", "MP", "PP"])
    hh = st.number_input("4. Enter number of households:", min_value=1, value=100)
    donated_pct = st.slider("5. Estimated % food donated:", 0, 100, 12)
    submitted = st.form_submit_button("Estimate Cost")

if submitted:
    with st.spinner("Calculating..."):
        df = load_data()
        model_dict, scores = train_model(df)
        miles = region_options[region_label]
        result = predict(model_dict, scores, program, region_label, hh, weight, donated_pct, miles)

    if result:
        st.markdown(f"""
        <div style='background-color: #fff; padding: 20px; border-radius: 8px; max-width: 500px; margin: auto; color: black;'>
        <h4 style='color: {FSD_GREEN};'>User Inputs</h4>
        <p><strong>Region:</strong> {region_label}</p>
        <p><strong>Program:</strong> {program}</p>
        <p><strong>% Donated:</strong> {donated_pct}%</p>
        <p><strong>Households:</strong> {hh}</p>
        <p><strong>Total Weight:</strong> {weight:.1f} lbs</p>
        <p><strong>Distance:</strong> {miles} miles</p>
        <hr style="border: none; border-top: 1px solid #ccc;">
        <h4 style='color: {FSD_ORANGE};'>Estimated Costs</h4>
        <p><strong>Quarterly Cost:</strong> ${result['Quarterly Cost']:,.2f}</p>
        <p><strong>Base Annual Cost:</strong> ${result['Base Annual Cost']:,.2f}</p>
        <p><strong>Fixed Cost:</strong> ${result['Fixed Cost']:,.2f}</p>
        <p><strong>Transport Cost:</strong> ${result['Transport Cost']:,.2f}</p>
        <p><strong>Total Cost:</strong> ${result['Total Cost']:,.2f}</p>
        <p><strong>Cost per lb:</strong> ${result['Cost per lb']:,.4f}</p>
        <p><strong>Upper Bound (MAE):</strong> ${result['Upper Bound MAE']:,.2f}</p>
        <p><strong>Upper Bound (RMSE):</strong> ${result['Upper Bound RMSE']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Prediction failed. Check inputs or data.")
