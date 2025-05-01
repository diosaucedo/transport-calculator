# Final Streamlit Calculator with Clean Output

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Transport Cost Calculator", layout="centered")
st.image("FSD LOGO.png", width=180)
st.title("Transport Cost Calculator")
st.subheader("Built for Feeding San Diego")

@st.cache_data
def load_data():
    quarter_df = pd.read_excel("Final_Quarterly.xlsx")
    merged_df = pd.read_excel("merged_final.xlsx")

    merged_df = merged_df[['Location', 'PROGRAM', '% Donated (per location)']]
    program_means = merged_df.groupby('PROGRAM')['% Donated (per location)'].mean().reset_index()
    program_means.rename(columns={'% Donated (per location)': 'Program_Mean_Donated_%'}, inplace=True)
    quarter_df = quarter_df.merge(merged_df, on=['Location', 'PROGRAM'], how='left')
    quarter_df = quarter_df.merge(program_means, on='PROGRAM', how='left')

    quarter_df['Estimated_Donated_%'] = (
        0.8 * quarter_df['% Donated (per location)'] + 0.2 * quarter_df['Program_Mean_Donated_%']
    )
    quarter_df['Estimated_Donated_Weight'] = quarter_df['Estimated_Donated_%'] * quarter_df['Weight']
    quarter_df['Estimated_Purchased_Weight'] = quarter_df['Weight'] - quarter_df['Estimated_Donated_Weight']

    programs = ['AGENCY', 'SP', 'BP', 'MP', 'PP']
    df = quarter_df[quarter_df['PROGRAM'].isin(programs) & (quarter_df['Cost'] > 1)].copy()
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
    df.loc[(df['PROGRAM'] == 'PP') & (df['Total_Weight'] < 150000), 'Weight_Tier'] = 'All'
    df = df[df['Weight_Tier'] != 'Unassigned']
    return df

@st.cache_resource
def train_models():
    df = load_data()
    features = ['Region', 'PROGRAM','Quarter', 'hh', 'Estimated_Donated_Weight', 'Estimated_Purchased_Weight']
    target = 'Cost'
    model_dict = {}

    for (program, tier), subset in df.groupby(['PROGRAM', 'Weight_Tier']):
        if len(subset) < 20:
            continue
        X = subset[features]
        y = np.log1p(subset[target])
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        pre = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Region','PROGRAM','Quarter']),
            ('num', 'passthrough', ['hh','Estimated_Donated_Weight','Estimated_Purchased_Weight'])
        ])

        pipe = Pipeline([
            ('preprocessor', pre),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200))
        ])

        pipe.fit(X_train, y_train)
        model_dict[(program, tier)] = pipe

    return model_dict

model_dict = train_models()

# === Input form ===
weight = st.number_input("1. What is the total weight in pounds?", min_value=0.0, step=1.0)
region = st.selectbox("2. Where is the delivery going?", ['North Coastal (27.7 mi)', 'North Inland (46.6 mi)', 'Central (19.2 mi)', 'East (60.5 mi)', 'South (28.3 mi)'])
agency = st.selectbox("3. Which agency/program is it?", ["AGENCY", "SP", "BP", "MP", "PP"])
hh = st.number_input("4. How many households are served?", min_value=1, step=1)
donated = st.slider("5. Estimated % Donated", 0, 100, 50)

if st.button("Calculate"):
    miles = float(region.split('(')[1].split(' ')[0])
    quarterly_total = weight / 4
    q_donated = quarterly_total * (donated / 100)
    q_purchased = quarterly_total * (1 - donated / 100)

    if agency == 'AGENCY':
        tier = 'Low' if quarterly_total < 8000 else 'Mid' if quarterly_total <= 20000 else 'High'
    elif agency == 'MP':
        tier = 'Low' if quarterly_total < 6000 else 'High'
    elif agency == 'SP':
        tier = 'Low' if quarterly_total < 2000 else 'High'
    elif agency in ['BP', 'PP']:
        tier = 'All'
    else:
        st.error("Invalid program")
        st.stop()

    model = model_dict.get((agency, tier))
    if not model:
        st.error("No model found for selection")
        st.stop()

    input_df = pd.DataFrame([{
        'Region': region,
        'PROGRAM': agency,
        'Quarter': 'Q1',
        'hh': hh,
        'Estimated_Donated_Weight': q_donated,
        'Estimated_Purchased_Weight': q_purchased
    }])

    pred = model.predict(input_df)[0]
    total_cost = np.expm1(pred) * 4
    cost_per_lb_per_mile = total_cost / (weight * miles)

    st.markdown(f'''
    <div style="background-color:#fff;padding:20px;border-radius:10px;max-width:350px;margin:auto">
    <h4 style="color:#3c763d;">Calculation Completed</h4>
    <p><strong style='color:#000;'>Agency:</strong> <span style='color:#000'>{agency}</span></p>
    <p><strong style='color:#000;'>Region:</strong> <span style='color:#000'>{region}</span></p>
    <p><strong style='color:#000;'>Weight:</strong> <span style='color:#000'>{weight:.1f} lbs</span></p>
    <p><strong style='color:#000;'>Households:</strong> <span style='color:#000'>{hh}</span></p>
    <p><strong style='color:#000;'>% Donated:</strong> <span style='color:#000'>{donated:.1f}%</span></p>
    <p><strong style='color:#000;'>Distance:</strong> <span style='color:#000'>{miles:.1f} miles</span></p>
    <hr style="border:none; border-top:1px solid #ccc;">
    <p style="color:#F7941D;font-size:16px;font-weight:bold;text-align:center">
        Cost per lb per mile: ${cost_per_lb_per_mile:.4f}
    </p>
    </div>
    ''', unsafe_allow_html=True)
