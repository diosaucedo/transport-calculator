import streamlit as st
import pandas as pd
import base64

# === Load logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_path = f"data:image/png;base64,{encoded_image}"

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ≈ 0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02
donated_cost = 0.04  # locked

# === Cost estimates from data ===
cost_estimates = pd.DataFrame([
    {'PROGRAM': 'AGENCY', 'estimated_produce_cost_per_lb': 0.079, 'estimated_purchased_cost_per_lb': 0.85},
    {'PROGRAM': 'BP', 'estimated_produce_cost_per_lb': 0.17, 'estimated_purchased_cost_per_lb': 1.27},
    {'PROGRAM': 'MEM', 'estimated_produce_cost_per_lb': 0.12, 'estimated_purchased_cost_per_lb': 0.91},
    {'PROGRAM': 'MP', 'estimated_produce_cost_per_lb': 0.11, 'estimated_purchased_cost_per_lb': 0.89},
    {'PROGRAM': 'PP', 'estimated_produce_cost_per_lb': 0.10, 'estimated_purchased_cost_per_lb': 0.93},
    {'PROGRAM': 'SP', 'estimated_produce_cost_per_lb': 0.13, 'estimated_purchased_cost_per_lb': 0.96}
])

# === Default lbs per HH
lbs_per_hh = {
    'AGENCY': {'produce': 4, 'purchased': 6, 'donated': 5},
    'MEM': {'produce': 3, 'purchased': 5, 'donated': 2},
    'BP': {'produce': 2, 'purchased': 7, 'donated': 1},
    'MP': {'produce': 3, 'purchased': 4, 'donated': 2},
    'PP': {'produce': 3, 'purchased': 6, 'donated': 2},
    'SP': {'produce': 5, 'purchased': 5, 'donated': 1}
}

# === Layout ===
st.markdown(f"<div style='text-align: center;'><img src='{logo_path}' style='height: 80px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
st.title("Delivery Cost Calculator")

with st.form("calculator_form"):
    program = st.selectbox("1. Which program is this?", list(lbs_per_hh.keys()))
    hh = st.number_input("2. How many households are served?", min_value=1, value=350)
    produce_lb = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=float(lbs_per_hh[program]['produce']))
    purchased_lb = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=float(lbs_per_hh[program]['purchased']))
    donated_lb = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=float(lbs_per_hh[program]['donated']))
    miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

    submitted = st.form_submit_button("Calculate & Estimate")

if submitted:
    # === Look up costs
    row = cost_estimates[cost_estimates['PROGRAM'] == program].iloc[0]
    produce_cost = row['estimated_produce_cost_per_lb']
    purchased_cost = row['estimated_purchased_cost_per_lb']

    # === Calculate weights
    prod_total = produce_lb * hh
    purch_total = purchased_lb * hh
    don_total = donated_lb * hh
    total_lbs = prod_total + purch_total + don_total

    # === Calculate costs
    base_cost = prod_total * produce_cost + purch_total * purchased_cost + don_total * donated_cost
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    # === Per HH itemized cost
    prod_cost_hh = produce_lb * produce_cost
    purch_cost_hh = purchased_lb * purchased_cost
    don_cost_hh = donated_lb * donated_cost

    # === Styled output
    st.markdown(f"""
    <div style="background-color: white; padding: 20px 30px; border-radius: 16px; max-width: 400px; margin: 0 auto; font-family: sans-serif;">
        <h3 style="color: #3c763d; margin-top: 0;">✅ Calculation Completed</h3>

        <h4 style="color: #6BA539; margin-bottom: 5px;">User Inputs</h4>
        <p><strong>Program:</strong> {program}</p>
        <p><strong>Households:</strong> {hh}</p>
        <p><strong>Produce per HH:</strong> {produce_lb}</p>
        <p><strong>Purchased per HH:</strong> {purchased_lb}</p>
        <p><strong>Donated per HH:</strong> {donated_lb}</p>
        <p><strong>Distance:</strong> {miles} miles</p>

        <hr style="margin: 12px 0;">

        <h4 style="color: #6BA539; margin-bottom: 5px;">Calculator Outputs</h4>
        <p><strong>Total Weight:</strong> {total_lbs:.2f} lbs</p>
        <p><strong>Base Food Cost:</strong> ${base_cost:.2f}</p>
        <p><strong>Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb):</strong> ${fixed_cost:.2f}</p>
        <p><strong>Transport Cost (@ $0.02/lb/mile):</strong> ${transport_cost:.2f}</p>

        <hr style="margin: 12px 0;">

        <h4 style="color: #6BA539; margin-bottom: 5px;">Food Cost Per lb Per HH</h4>
        <p><strong>Produce:</strong> {produce_lb} lbs × ${produce_cost:.3f} = ${prod_cost_hh:.2f} per HH</p>
        <p><strong>Purchased:</strong> {purchased_lb} lbs × ${purchased_cost:.3f} = ${purch_cost_hh:.2f} per HH</p>
        <p><strong>Donated:</strong> {donated_lb} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f} per HH</p>

        <hr style="margin: 12px 0;">

        <h4 style="color: #6BA539; margin-bottom: 5px;">Final Outputs</h4>
        <p><strong>Delivery Cost (Food + Transport):</strong> ${delivery_cost:.2f}</p>
        <p><strong>Total Cost:</strong> ${total_cost:.2f}</p>
        <p><strong>Blended Cost per lb:</strong> ${total_cost / total_lbs:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
