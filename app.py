# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
logo_path = f"data:image/png;base64,{encoded_image}"

# === Load Data and Train Models ===
@st.cache_data
def load_and_train_models():
    df = pd.read_excel("Final_Quarterly.xlsx")
    df['Total_Weight'] = df['Estimated_Donated_Weight'] + df['Estimated_Purchased_Weight']
    df['Weight_Tier'] = 'Unassigned'

    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] < 8000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] >= 8000) & (df['Total_Weight'] <= 20000), 'Weight_Tier'] = 'Mid'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] > 20000) & (df['Total_Weight'] < 150000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'BP') & (df['Total_Weight'] < 7311), 'Weight_Tier'] = 'All'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] < 6000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] >= 6000) & (df['Total_Weight'] < 60865), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] < 2000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] >= 2000) & (df['Total_Weight'] < 43175.75), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'PP') & (df['Total_Weight'] < 150000), 'Weight_Tier'] = 'All'

    df = df[df['Weight_Tier'] != 'Unassigned']

    features = ['Region', 'PROGRAM', 'Quarter', 'hh', 'Estimated_Donated_Weight', 'Estimated_Purchased_Weight']
    target = 'Cost'
    cat_cols = ['Region', 'PROGRAM', 'Quarter']
    num_cols = [col for col in features if col not in cat_cols]

    model_dict = {}
    score_summary = []

    for (program, tier), subset in df.groupby(['PROGRAM', 'Weight_Tier']):
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
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        })

    score_df = pd.DataFrame(score_summary)
    return model_dict, score_df

# === Prediction Function ===
def predict_total_annual_cost(program, region, hh, total_weight, donated_pct, miles, model_dict, score_df):
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
    if not model:
        return None

    row = score_df[(score_df['Program'] == program) & (score_df['Weight_Tier'] == tier)]
    if row.empty:
        return None

    mae = row['MAE'].values[0]
    rmse = row['RMSE'].values[0]

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

    return {
        'Program': program,
        'Region': region,
        'Donated %': donated_pct,
        'Quarterly Cost': round(base_cost / 4, 2),
        'Base Annual Cost': round(base_cost, 2),
        'Fixed Cost Allocated': round(fixed_cost, 2),
        'Transportation Cost': round(transport_cost, 2),
        'Total Cost': round(total_cost, 2),
        'Cost per lb': round(cost_per_lb, 4),
        'Upper Bound MAE': round(total_cost + mae * 4, 2),
        'Upper Bound RMSE': round(total_cost + rmse * 4, 2)
    }

# === UI ===
st.set_page_config(page_title="Cost Estimator", layout="centered")
st.markdown(f"<div style='text-align: center;'><img src='{logo_path}' style='height: 110px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Annual Food Program Cost Estimator</h2>", unsafe_allow_html=True)

region_options = {
    'North Coastal (27.7 mi)': 27.7,
    'North Inland (46.6 mi)': 46.6,
    'Central (19.2 mi)': 19.2,
    'East (60.5 mi)': 60.5,
    'South (28.3 mi)': 28.3
}

with st.form("calculator"):
    st.markdown("### 1. Enter total weight (lbs):")
    weight = st.number_input("", min_value=1.0, value=10000.0)
    st.markdown("### 2. Select delivery region:")
    region_label = st.selectbox("", options=list(region_options.keys()))
    st.markdown("### 3. Select agency/program:")
    program = st.selectbox("", options=["AGENCY", "SP", "BP", "MP", "PP"])
    st.markdown("### 4. Enter number of households:")
    hh = st.number_input("", min_value=1, value=100)
    st.markdown("### 5. Estimated % food donated:")
    donated_pct = st.slider("", 0, 100, 50)

    submitted = st.form_submit_button("Calculate & Predict")

if submitted:
    with st.spinner("Calculating..."):
        model_dict, score_df = load_and_train_models()
        miles = region_options[region_label]
        result = predict_total_annual_cost(program, region_label, hh, weight, donated_pct, miles, model_dict, score_df)

    if result:
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; display: inline-block; text-align: left; max-width: 500px; color: black;">
                <h4 style="color: {FSD_GREEN}; margin-top: 0;">User Inputs</h4>
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
                <p><strong>Total Cost:</strong> ${result['Total Cost']:,.2f}</p>
                <p><strong>Cost per lb:</strong> ${result['Cost per lb']:,.4f}</p>
                <p><strong>Upper Bound (MAE):</strong> ${result['Upper Bound MAE']:,.2f}</p>
                <p><strong>Upper Bound (RMSE):</strong> ${result['Upper Bound RMSE']:,.2f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Prediction failed. Please check your inputs.")
