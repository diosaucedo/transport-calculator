import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Transport Cost Calculator", layout="centered")
st.title("Transport Cost Calculator")

@st.cache_resource
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
def train_models(df):
    model_dict = {}
    features = ['Region', 'PROGRAM','Quarter', 'hh', 'Estimated_Donated_Weight', 'Estimated_Purchased_Weight']
    target = 'Cost'

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

def forecast_cost_per_lb_mile(model_dict, program, region, hh, total_weight, donated_pct, miles):
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
    if model is None: return None

    input_df = pd.DataFrame([{
        'Region': region,
        'PROGRAM': program,
        'Quarter': 'Q1',
        'hh': hh,
        'Estimated_Donated_Weight': q_donated,
        'Estimated_Purchased_Weight': q_purchased
    }])

    pred = model.predict(input_df)[0]
    total_cost = np.expm1(pred) * 4
    return total_cost / (total_weight * miles)

# Load and train
with st.spinner("Loading model..."):
    df = load_data()
    model_dict = train_models(df)

# UI inputs
st.header("Enter Delivery Info")
region_label_to_value = {
    'North Coastal (27.7 mi)': 27.7,
    'North Inland (46.6 mi)': 46.6,
    'Central (19.2 mi)': 19.2,
    'East (60.5 mi)': 60.5,
    'South (28.3 mi)': 28.3
}
region_label = st.selectbox("Delivery Region", list(region_label_to_value.keys()))
distance = region_label_to_value[region_label]

weight = st.number_input("Total Weight (lbs)", min_value=0.0)
hh = st.number_input("Households Served", min_value=1)
agency = st.selectbox("Program/Agency", ["AGENCY", "SP", "BP", "MP", "PP"])
donated_pct = st.slider("Estimated % Donated", 0, 100, 50)

if st.button("Calculate Cost"):
    if weight > 0 and hh > 0:
        cost = forecast_cost_per_lb_mile(model_dict, agency, region_label, hh, weight, donated_pct, distance)
        if cost:
            st.success(f"Cost per lb per mile: ${cost:.4f}")
        else:
            st.error("No model available for this program and tier.")
    else:
        st.warning("Please enter valid weight and households.")
