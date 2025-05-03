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

# === Load Data ===
@st.cache_data
def load_data():
    merged_df = pd.read_excel("merged_final.xlsx")
    quarter_df = pd.read_excel("Final_Quarterly.xlsx")

    merged_df = merged_df[['Location', 'PROGRAM', '% Donated (per location)']]
    program_means = merged_df.groupby('PROGRAM')['% Donated (per location)'].mean().reset_index()
    program_means.rename(columns={'% Donated (per location)': 'Program_Mean_Donated_%'}, inplace=True)

    quarter_df = quarter_df.merge(merged_df, on=['Location', 'PROGRAM'], how='left')
    quarter_df = quarter_df.merge(program_means, on='PROGRAM', how='left')

    quarter_df['Estimated_Donated_%'] = 0.8 * quarter_df['% Donated (per location)'] + 0.2 * quarter_df['Program_Mean_Donated_%']
    quarter_df['Estimated_Donated_Weight'] = quarter_df['Estimated_Donated_%'] * quarter_df['Weight']
    quarter_df['Estimated_Purchased_Weight'] = quarter_df['Weight'] - quarter_df['Estimated_Donated_Weight']
    quarter_df['Total_Weight'] = quarter_df['Estimated_Donated_Weight'] + quarter_df['Estimated_Purchased_Weight']

    df = quarter_df[quarter_df['Cost'] > 1].copy()
    df['Weight_Tier'] = 'Unassigned'

    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] < 8000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] >= 8000) & (df['Total_Weight'] <= 20000), 'Weight_Tier'] = 'Mid'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] > 20000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'BP') & (df['Total_Weight'] < 7311), 'Weight_Tier'] = 'All'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] < 6000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] >= 6000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] < 2000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] >= 2000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'PP') & (df['Total_Weight'] < 150000), 'Weight_Tier'] = 'All'

    return df[df['Weight_Tier'] != 'Unassigned']

# === Train Models ===
def train_models(df):
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pre = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ])

        pipe = Pipeline([
            ('preprocessor', pre),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200))
        ])
        pipe.fit(X_train, y_train)

        y_pred = np.expm1(pipe.predict(X_test))
        y_actual = np.expm1(y_test)

        model_dict[(program, tier)] = pipe
        scores.append({
            'Program': program,
            'Weight_Tier': tier,
            'MAE': mean_absolute_error(y_actual, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_actual, y_pred))
        })

    return model_dict, pd.DataFrame(scores)

# === Predict ===
def predict(model_dict, score_df, program, region, hh, weight, donated_pct, miles):
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

    row = score_df[(score_df['Program'] == program) & (score_df['Weight_Tier'] == tier)]
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
st.set_page_config(page_title="FSD Cost Estimator", layout="centered")
st.image("FSD LOGO.png", width=160)
st.title("Cost Estimator")

region_options = {
    'North Coastal (27.7 mi)': 27.7,
    'North Inland (46.6 mi)': 46.6,
    'Central (19.2 mi)': 19.2,
    'East (60.5 mi)': 60.5,
    'South (28.3 mi)': 28.3
}

with st.form("estimator"):
    st.markdown("### 1. Enter total weight (lbs):")
    weight = st.number_input("", min_value=1.0, value=2000.0)

    st.markdown("### 2. Select delivery region:")
    region_label = st.selectbox("", list(region_options.keys()))

    st.markdown("### 3. Select agency/program:")
    program = st.selectbox("", ["AGENCY", "SP", "BP", "MP", "PP"])

    st.markdown("### 4. Enter number of households:")
    hh = st.number_input("", min_value=1, value=100)

    st.markdown("### 5. Estimated % food donated:")
    donated_pct = st.slider("% Donated", 0, 100, 30)

    submitted = st.form_submit_button("Estimate Cost")

if submitted:
    with st.spinner("Loading models and calculating..."):
        df = load_data()
        models, scores = train_models(df)
        miles = region_options[region_label]
        result = predict(models, scores, program, region_label, hh, weight, donated_pct, miles)

    if result:
        st.markdown(f"""
        <div style='background-color: #fff; color: black; padding: 20px; border-radius: 8px; max-width: 500px; margin: auto;'>
        <h4 style='color: {FSD_GREEN};'>User Inputs</h4>
        <p><strong>Region:</strong> {region_label}</p>
        <p><strong>Program:</strong> {program}</p>
        <p><strong>% Donated:</strong> {donated_pct}%</p>
        <p><strong>Households:</strong> {hh}</p>
        <p><strong>Total Weight:</strong> {weight:.1f} lbs</p>
        <p><strong>Distance:</strong> {miles} miles</p>
        <hr>
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
        st.error("Prediction failed. Check inputs or try again.")
