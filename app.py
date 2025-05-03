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

# === Logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_html = f"<img src='data:image/png;base64,{encoded_image}' style='height: 110px; margin-bottom: 20px;'>"

# === Load data ===
merged_final_df = pd.read_excel("merged_final.xlsx")
quarter_df = pd.read_excel("Final_Quarterly.xlsx")

merged_df = merged_final_df[['Location', 'PROGRAM', '% Donated (per location)']]
program_means = merged_df.groupby('PROGRAM')['% Donated (per location)'].mean().reset_index()
program_means.rename(columns={'% Donated (per location)': 'Program_Mean_Donated_%'}, inplace=True)

quarter_df = quarter_df.merge(merged_df, on=['Location', 'PROGRAM'], how='left')
quarter_df = quarter_df.merge(program_means, on='PROGRAM', how='left')

quarter_df['Estimated_Donated_%'] = (
    0.8 * quarter_df['% Donated (per location)'] +
    0.2 * quarter_df['Program_Mean_Donated_%']
)

quarter_df['Estimated_Donated_Weight'] = quarter_df['Estimated_Donated_%'] * quarter_df['Weight']
quarter_df['Estimated_Purchased_Weight'] = quarter_df['Weight'] - quarter_df['Estimated_Donated_Weight']

programs_to_include = ['AGENCY', 'SP', 'BP', 'MP', 'PP']
quarter_df_filtered = quarter_df[quarter_df['PROGRAM'].isin(programs_to_include)].copy()
df_filtered = quarter_df_filtered[quarter_df_filtered['Cost'] > 1].copy()

# === Assign weight tiers ===
df_filtered['Total_Weight'] = df_filtered['Estimated_Donated_Weight'] + df_filtered['Estimated_Purchased_Weight']
df_filtered['Weight_Tier'] = 'Unassigned'

df_filtered.loc[(df_filtered['PROGRAM'] == 'AGENCY') & (df_filtered['Total_Weight'] < 8000), 'Weight_Tier'] = 'Low'
df_filtered.loc[(df_filtered['PROGRAM'] == 'AGENCY') & (df_filtered['Total_Weight'] >= 8000) & (df_filtered['Total_Weight'] <= 20000), 'Weight_Tier'] = 'Mid'
df_filtered.loc[(df_filtered['PROGRAM'] == 'AGENCY') & (df_filtered['Total_Weight'] > 20000) & (df_filtered['Total_Weight'] < 150000), 'Weight_Tier'] = 'High'
df_filtered.loc[(df_filtered['PROGRAM'] == 'BP') & (df_filtered['Total_Weight'] < 7311), 'Weight_Tier'] = 'All'
df_filtered.loc[(df_filtered['PROGRAM'] == 'MP') & (df_filtered['Total_Weight'] < 6000), 'Weight_Tier'] = 'Low'
df_filtered.loc[(df_filtered['PROGRAM'] == 'MP') & (df_filtered['Total_Weight'] >= 6000) & (df_filtered['Total_Weight'] < 60865), 'Weight_Tier'] = 'High'
df_filtered.loc[(df_filtered['PROGRAM'] == 'SP') & (df_filtered['Total_Weight'] < 2000), 'Weight_Tier'] = 'Low'
df_filtered.loc[(df_filtered['PROGRAM'] == 'SP') & (df_filtered['Total_Weight'] >= 2000) & (df_filtered['Total_Weight'] < 43175.75), 'Weight_Tier'] = 'High'
df_filtered.loc[(df_filtered['PROGRAM'] == 'PP') & (df_filtered['Total_Weight'] < 150000), 'Weight_Tier'] = 'All'
df_filtered = df_filtered[df_filtered['Weight_Tier'] != 'Unassigned']

# === Train models ===
features = ['Region', 'PROGRAM', 'Quarter', 'hh', 'Estimated_Donated_Weight', 'Estimated_Purchased_Weight']
target = 'Cost'
cat_cols = ['Region', 'PROGRAM', 'Quarter']
num_cols = [col for col in features if col not in cat_cols]

model_dict = {}
score_summary = []

for (program, tier), subset in df_filtered.groupby(['PROGRAM', 'Weight_Tier']):
    if len(subset) < 20:
        continue
    X = subset[features]
    y = np.log1p(subset[target])
    mask = (~y.isna()) & (~np.isinf(y))
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=5, learning_rate=0.1))
    ])

    pipeline.fit(X_train, y_train)
    model_dict[(program, tier)] = pipeline
    y_pred = np.expm1(pipeline.predict(X_test))
    y_true = np.expm1(y_test)
    score_summary.append({
        'Program': program,
        'Weight_Tier': tier,
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'MAE': round(mean_absolute_error(y_true, y_pred), 2)
    })

summary_df = pd.DataFrame(score_summary)

# === Prediction Function ===
def predict_total_annual_cost(program, region, hh, total_weight, donated_pct, miles):
    donated_ratio = donated_pct / 100.0
    quarterly_total = total_weight / 4
    q_donated = quarterly_total * donated_ratio
    q_purchased = quarterly_total * (1 - donated_ratio)

    if program == 'AGENCY':
        tier = 'Low' if quarterly_total < 8000 else 'Mid' if quarterly_total <= 20000 else 'High'
    elif program == 'MP':
        tier = 'Low' if quarterly_total < 6000 else 'High'
    elif program == 'SP':
        tier = 'Low' if quarterly_total < 2000 else 'High'
    elif program in ['BP', 'PP']:
        tier = 'All'
    else:
        return None

    model = model_dict.get((program, tier))
    if model is None:
        return None

    input_df = pd.DataFrame([{
        'Region': region,
        'PROGRAM': program,
        'Quarter': 'Q1',
        'hh': hh,
        'Estimated_Donated_Weight': q_donated,
        'Estimated_Purchased_Weight': q_purchased
    }])

    pred_log = model.predict(input_df)[0]
    base_cost = np.expm1(pred_log) * 4
    fixed_cost = total_weight * FIXED_COST_PER_LB
    transport_cost = total_weight * miles * TRANSPORT_RATE
    total_cost = base_cost + fixed_cost + transport_cost
    cost_per_lb = total_cost / total_weight

    row = summary_df[(summary_df['Program'] == program) & (summary_df['Weight_Tier'] == tier)]
    mae = row['MAE'].values[0]
    rmse = row['RMSE'].values[0]

    return {
        'Program': program,
        'Region': region,
        'Donated %': donated_pct,
        'Quarterly Cost': round(base_cost / 4, 2),
        'Base Annual Cost': round(base_cost, 2),
        'Fixed Cost Allocated': round(fixed_cost, 2),
        'Transportation Cost': round(transport_cost, 2),
        'Total Cost (with fixed & transport)': round(total_cost, 2),
        'Cost per lb': round(cost_per_lb, 4),
        'Upper Bound MAE': round(total_cost + mae * 4, 2),
        'Upper Bound RMSE': round(total_cost + rmse * 4, 2)
    }

# === Streamlit UI ===
st.set_page_config(page_title="Cost Estimator", layout="centered")
st.markdown(f"<div style='text-align: center;'>{logo_html}</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Cost Estimator</h2>", unsafe_allow_html=True)

with st.form("calculator_form"):
    weight = st.number_input("1. Enter total weight (lbs):", min_value=0.0, step=100.0)
    region_map = {
        'North Coastal (27.7 mi)': 27.7,
        'North Inland (46.6 mi)': 46.6,
        'Central (19.2 mi)': 19.2,
        'East (60.5 mi)': 60.5,
        'South (28.3 mi)': 28.3
    }
    region_label = st.selectbox("2. Select delivery region:", list(region_map.keys()))
    miles = region_map[region_label]
    program = st.selectbox("3. Select agency/program:", ["AGENCY", "SP", "BP", "MP", "PP"])
    hh = st.number_input("4. Enter number of households:", min_value=0, step=1)
    donated = st.slider("5. Estimated % food donated:", 0, 100, 50)

    submitted = st.form_submit_button("Calculate")

if submitted:
    result = predict_total_annual_cost(program, region_label, hh, weight, donated, miles)

    if not result:
        st.error("Prediction failed.")
    else:
        st.markdown(f"""
        <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; color: black; max-width: 500px; margin: auto;">
            <h4 style="color: {FSD_GREEN};">User Inputs</h4>
            <p><strong>Program:</strong> {result['Program']}</p>
            <p><strong>Region:</strong> {result['Region']}</p>
            <p><strong>% Donated:</strong> {result['Donated %']}%</p>
            <p><strong>Households:</strong> {hh}</p>
            <p><strong>Total Weight:</strong> {weight:.1f} lbs</p>
            <p><strong>Distance:</strong> {miles} mi</p>
            <hr style="border: none; border-top: 1px solid #ccc;">
            <h4 style="color: {FSD_ORANGE};">Estimated Costs</h4>
            <p><strong>Quarterly Cost:</strong> ${result['Quarterly Cost']:,.2f}</p>
            <p><strong>Base Annual Cost:</strong> ${result['Base Annual Cost']:,.2f}</p>
            <p><strong>Allocated Fixed Cost:</strong> ${result['Fixed Cost Allocated']:,.2f}</p>
            <p><strong>Transportation Cost (@ $0.02/lb/mi):</strong> ${result['Transportation Cost']:,.2f}</p>
            <p><strong>Total Cost (with fixed & transport):</strong> ${result['Total Cost (with fixed & transport)']:,.2f}</p>
            <p><strong>Cost per lb:</strong> ${result['Cost per lb']:,.4f}</p>
            <p><strong>Upper Bound (MAE):</strong> ${result['Upper Bound MAE']:,.2f}</p>
            <p><strong>Upper Bound (RMSE):</strong> ${result['Upper Bound RMSE']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
