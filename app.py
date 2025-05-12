# === Install & Run Commands (run these in terminal) ===
# pip install streamlit gspread oauth2client pandas numpy openpyxl
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# === Google Sheets Setup ===
def get_gsheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    return gspread.authorize(creds)

# === Load logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_path = f"data:image/png;base64,{encoded_image}"

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606
TRANSPORT_COST_PER_LB_PER_MILE = 0.02
DONATED_COST = 0.04

# === Load model input data and compute cost estimates ===
quarter_df = pd.read_excel('Final_Quarterly_Data.xlsx')
merged_df = pd.read_excel('merged_final.xlsx')

merged_df = merged_df[['Location', 'PROGRAM', '% Donated (per location)']]
program_means = merged_df.groupby('PROGRAM')['% Donated (per location)'].mean().reset_index()
program_means.rename(columns={'% Donated (per location)': 'Program_Mean_Donated_%'}, inplace=True)

final_df = quarter_df.merge(merged_df, on=['Location', 'PROGRAM'], how='left')
final_df = final_df.merge(program_means, on='PROGRAM', how='left')
final_df['Estimated_Donated_%'] = (
    0.8 * final_df['% Donated (per location)'] +
    0.2 * final_df['Program_Mean_Donated_%']
)
final_df['Estimated_Donated_Weight'] = final_df['Estimated_Donated_%'] * final_df['Weight']
final_df['Estimated_Purchased_Weight'] = final_df['Weight'] - final_df['Estimated_Donated_Weight']

lbs_per_hh_model = {
    'AGENCY': {'produce': 16, 'purchased': None, 'donated': None},
    'BP': {'produce': 4, 'purchased': 4, 'donated': 4},
    'MP': {'produce': 16, 'purchased': 5, 'donated': 2},
    'PP': {'produce': 24, 'purchased': 0, 'donated': 0},
    'SP': {'produce': 16, 'purchased': 5, 'donated': 2}
}

program_food_ratios = {}
for prog, values in lbs_per_hh_model.items():
    total = sum(v for v in values.values() if v is not None)
    program_food_ratios[prog] = {
        'produce_ratio': (values['produce'] or 0) / total if total else 0,
        'purchased_ratio': (values['purchased'] or 0) / total if total else 0,
        'donated_ratio': (values['donated'] or 0) / total if total else 0
    }

def apply_ratios(row):
    ratios = program_food_ratios.get(row['PROGRAM'], {'produce_ratio': 0, 'purchased_ratio': 0, 'donated_ratio': 0})
    weight = row['Weight']
    return pd.Series({
        'Estimated_Produce_Weight': weight * ratios['produce_ratio'],
        'Estimated_Purchased_Weight': weight * ratios['purchased_ratio'],
        'Estimated_Donated_Weight': weight * ratios['donated_ratio']
    })

estimated_weights = final_df.apply(apply_ratios, axis=1)
final_df[['Estimated_Produce_Weight', 'Estimated_Purchased_Weight', 'Estimated_Donated_Weight']] = estimated_weights

final_df = final_df[final_df['Cost'] > 1]
program_agg = final_df.groupby('PROGRAM').agg({
    'Cost': 'sum',
    'Weight': 'sum',
    'Estimated_Produce_Weight': 'sum',
    'Estimated_Purchased_Weight': 'sum',
    'Estimated_Donated_Weight': 'sum'
}).reset_index()

results = []
for _, row in program_agg.iterrows():
    prog = row['PROGRAM']
    total_cost = row['Cost']
    total_weight = row['Weight']
    prod_wt = row['Estimated_Produce_Weight']
    purch_wt = row['Estimated_Purchased_Weight']
    don_wt = row['Estimated_Donated_Weight']

    blended_cost = total_cost / total_weight if total_weight else 0
    rprod = prod_wt / total_weight if total_weight else 0
    rpurch = purch_wt / total_weight if total_weight else 0
    rdon = don_wt / total_weight if total_weight else 0

    x = y = None
    if prog != 'BP' and (rprod > 0 and rpurch > 0):
        candidate_ys = [round(v, 3) for v in np.linspace(0.5, 1.2, 71)]
        best_error = float('inf')
        for candidate_y in candidate_ys:
            candidate_x = (blended_cost - rpurch * candidate_y - rdon * DONATED_COST) / rprod
            reconstructed = rprod * candidate_x + rpurch * candidate_y + rdon * DONATED_COST
            error = abs(reconstructed - blended_cost)
            if error < best_error:
                best_error = error
                x, y = candidate_x, candidate_y
    elif rpurch > 0 and rprod == 0:
        y = (blended_cost - rdon * DONATED_COST) / rpurch
        y = max(0.5, min(1.2, y))
        x = None
    elif rprod > 0 and rpurch == 0:
        x = (blended_cost - rdon * DONATED_COST) / rprod
        y = None

    results.append({
        'PROGRAM': prog,
        'estimated_produce_cost_per_lb': x,
        'estimated_purchased_cost_per_lb': y
    })

cost_estimates = pd.DataFrame(results)

# === Streamlit UI ===
st.markdown(f"<div style='text-align: center;'><img src='{logo_path}' style='height: 140px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
st.title("Cost Calculator")

with st.form("calculator_form"):
    program = st.selectbox("1. Which program is this?", list(lbs_per_hh_model.keys()))
    hh = st.number_input("2. How many households are served?", min_value=1, value=350)
    produce_lb = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=0.0)
    purchased_lb = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=0.0)
    donated_lb = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=0.0)
    miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

    submitted = st.form_submit_button("Calculate & Estimate")

if submitted:
    match = cost_estimates[cost_estimates['PROGRAM'] == program]
    if match.empty:
        produce_cost = 0
        purchased_cost = 0
    else:
        row = match.iloc[0]
        produce_cost = row['estimated_produce_cost_per_lb'] if pd.notna(row['estimated_produce_cost_per_lb']) else 0
        purchased_cost = 1.27 if program == 'BP' else row['estimated_purchased_cost_per_lb'] if pd.notna(row['estimated_purchased_cost_per_lb']) else 0

    prod_total = produce_lb * hh
    purch_total = purchased_lb * hh
    don_total = donated_lb * hh
    total_lbs = prod_total + purch_total + don_total

    base_cost = prod_total * produce_cost + purch_total * purchased_cost + don_total * DONATED_COST
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    blended_cost_per_lb = total_cost / total_lbs if total_lbs else 0

    # === Display Results ===
    st.success(f"""
### Calculation Completed

**Total Weight:** {total_lbs:.2f} lbs  
**Base Food Cost:** ${base_cost:.2f}  
**Fixed Cost:** ${fixed_cost:.2f}  
**Transport Cost:** ${transport_cost:.2f}  
**Delivery Cost (Food + Transport):** ${delivery_cost:.2f}  
**Total Cost:** ${total_cost:.2f}  
**Blended Cost per lb:** ${blended_cost_per_lb:.4f}  
""")

    # === Append to Google Sheet ===
    try:
        client = get_gsheet_client()
        sheet = client.open("Streamlit Calculator Outputs").sheet1
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            program,
            hh,
            produce_lb,
            purchased_lb,
            donated_lb,
            miles,
            round(total_lbs, 2),
            round(base_cost, 2),
            round(fixed_cost, 2),
            round(transport_cost, 2),
            round(delivery_cost, 2),
            round(total_cost, 2),
            round(blended_cost_per_lb, 4)
        ])
        st.info("Logged successfully to Google Sheets.")
    except Exception as e:
        st.error(f"Google Sheets Logging failed: {e}")
