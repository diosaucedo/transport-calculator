import streamlit as st
import pandas as pd
import numpy as np
import base64
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === Fixed cost ===
FIXED_COST = 2607

# === Load and display logo ===
def load_logo():
    with open("FSD LOGO.png", "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded_image}' style='height:80px;margin-bottom:20px;'>"

# === Model Training ===
@st.cache_resource
def load_and_train_model():
    final_quarterly_path = 'Final_Quarterly.xlsx'
    merged_final_path = 'merged_final.xlsx'

    quarter_df = pd.read_excel(final_quarterly_path)
    merged_df = pd.read_excel(merged_final_path)

    merged_df = merged_df[['Location', 'PROGRAM', '% Donated (per location)']]
    program_means = merged_df.groupby('PROGRAM')['% Donated (per location)'].mean().reset_index()
    program_means.rename(columns={'% Donated (per location)': 'Program_Mean_Donated_%'}, inplace=True)
    quarter_df = quarter_df.merge(merged_df, on=['Location', 'PROGRAM'], how='left')
    quarter_df = quarter_df.merge(program_means, on='PROGRAM', how='left')

    quarter_df['Estimated_Donated_%'] = (
        0.8 * quarter_df['% Donated (per location)'] + 0.2 * quarter_df['Program_Mean_Donated_%']
    )

    programs = ['AGENCY', 'SP', 'BP', 'MP', 'PP']
    df = quarter_df[quarter_df['PROGRAM'].isin(programs) & (quarter_df['Cost'] > 1)].copy()
    df['Weight_Tier'] = 'Unassigned'
    df['Total_Weight'] = df['Weight']

    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] < 8000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] >= 8000) & (df['Total_Weight'] <= 20000), 'Weight_Tier'] = 'Mid'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] > 20000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'BP') & (df['Total_Weight'] < 7311), 'Weight_Tier'] = 'All'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] < 6000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] >= 6000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] < 2000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] >= 2000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'PP') & (df['Total_Weight'] < 150000), 'Weight_Tier'] = 'All'
    df = df[df['Weight_Tier'] != 'Unassigned']

    features = ['Region', 'PROGRAM', 'Quarter', 'hh', 'Weight', 'Estimated_Donated_%']
    target = 'Cost'

    model_dict = {}
    for (program, tier), subset in df.groupby(['PROGRAM', 'Weight_Tier']):
        if len(subset) < 20: continue
        X = subset[features]
        y = np.log1p(subset[target])
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        pre = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Region', 'PROGRAM', 'Quarter']),
            ('num', 'passthrough', ['hh', 'Weight', 'Estimated_Donated_%'])
        ])

        pipe = Pipeline([
            ('preprocessor', pre),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200))
        ])

        pipe.fit(X_train, y_train)
        model_dict[(program, tier)] = pipe

    return model_dict

model_dict = load_and_train_model()

# === Forecast function ===
def forecast_cost_per_lb_mile(program, region, hh, total_weight, donated_pct, miles):
    donated_ratio = donated_pct / 100.0
    quarterly_total = total_weight / 4

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
    if model is None: return None

    input_df = pd.DataFrame([{
        'Region': region,
        'PROGRAM': program,
        'Quarter': 'Q1',
        'hh': hh,
        'Weight': total_weight,
        'Estimated_Donated_%': donated_ratio
    }])

    pred = model.predict(input_df)[0]
    base_cost = np.expm1(pred) * 4
    total_cost = base_cost + FIXED_COST
    cost_per_lb_per_mile = total_cost / (total_weight * miles)
    return cost_per_lb_per_mile, total_cost

# === Streamlit UI ===
st.markdown(load_logo(), unsafe_allow_html=True)
st.header("Cost per Pound per Mile Calculator")

weight = st.number_input("1. What is the total weight in pounds?", min_value=0.0, step=100.0)
region_display_to_miles = {
    'North Coastal (27.7 mi)': 27.7,
    'North Inland (46.6 mi)': 46.6,
    'Central (19.2 mi)': 19.2,
    'East (60.5 mi)': 60.5,
    'South (28.3 mi)': 28.3
}
region_label = st.selectbox("2. Where is the delivery going?", list(region_display_to_miles.keys()))
distance = region_display_to_miles[region_label]

agency = st.selectbox("3. Which agency/program is it?", ["AGENCY", "SP", "BP", "MP", "PP"])
hh = st.number_input("4. How many households are served?", min_value=0, step=1)
donated = st.slider("5. Estimated % Donated", 0, 100, 50)

if st.button("Calculate & Predict"):
    if weight <= 0 or hh <= 0:
        st.warning("⚠️ Enter valid weight and households.")
    else:
        result = forecast_cost_per_lb_mile(agency, region_label, hh, weight, donated, distance)
        if result is None:
            st.error("⚠️ No model available for selected combination.")
        else:
            pred_cost_per_lb_mile, total_cost = result
            st.markdown(f"""
                <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; max-width: 400px; border: 1px solid #ccc;">
                    <h4 style="color: #3c763d;">Calculation Completed</h4>
                    <p><strong>Agency:</strong> {agency}</p>
                    <p><strong>Region:</strong> {region_label}</p>
                    <p><strong>Weight:</strong> {weight:.1f} lbs</p>
                    <p><strong>Households:</strong> {hh}</p>
                    <p><strong>% Donated:</strong> {donated:.1f}%</p>
                    <p><strong>Distance:</strong> {distance} miles</p>
                    <hr>
                    <p style="text-align: center; color: #F7941D; font-size: 16px; font-weight: bold;">
                        Cost per lb per mile: ${pred_cost_per_lb_mile:.4f}
                    </p>
                    <p style="text-align: center; color: #3c763d; font-size: 16px; font-weight: bold;">
                        Total cost of delivery: ${total_cost:,.2f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
