import streamlit as st
import pandas as pd
import numpy as np
import base64
import textwrap  # 🔧 For fixing HTML indentation

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606
TRANSPORT_COST_PER_LB_PER_MILE = 0.02

cost_per_lb_fixed = {
    'produce': 0.17,
    'purchased': 0.85,
    'donated': 0.04,
    'purchased_bp': 1.27
}

lbs_per_hh = {
    'AGENCY': {'produce': 4, 'purchased': 6, 'donated': 5},
    'BP': {'produce': 2, 'purchased': 7, 'donated': 1}
}

# === Load logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_html = f"<img src='data:image/png;base64,{encoded_image}' style='height: 80px; margin-bottom: 20px;'>"

# === Streamlit UI ===
st.set_page_config(page_title="Delivery Cost Estimator", layout="centered")
st.markdown(f"<div style='text-align: center;'>{logo_html}</div>", unsafe_allow_html=True)

st.header("Food Delivery Cost Calculator")

program = st.selectbox("1. Which program is this?", options=list(lbs_per_hh.keys()))
hh = st.number_input("2. How many households are served?", min_value=1, value=350)
produce_per_hh = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=float(lbs_per_hh[program]['produce']))
purchased_per_hh = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=float(lbs_per_hh[program]['purchased']))
donated_per_hh = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=float(lbs_per_hh[program]['donated']))
miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

# === Calculation ===
if st.button("Calculate & Estimate"):
    prod_cost = cost_per_lb_fixed['produce']
    don_cost = cost_per_lb_fixed['donated']
    purch_cost = cost_per_lb_fixed['purchased_bp'] if program == 'BP' else cost_per_lb_fixed['purchased']

    prod_total = produce_per_hh * hh
    purch_total = purchased_per_hh * hh
    don_total = donated_per_hh * hh
    total_lbs = prod_total + purch_total + don_total

    base_cost = prod_total * prod_cost + purch_total * purch_cost + don_total * don_cost
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    prod_cost_hh = produce_per_hh * prod_cost
    purch_cost_hh = purchased_per_hh * purch_cost
    don_cost_hh = donated_per_hh * don_cost

    # ✅ HTML Output using dedent to avoid rendering issues
    result_html = textwrap.dedent(f"""<div style='text-align: center;'>
    <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; display: inline-block; text-align: left; max-width: 360px;">
        <h3 style="color: #3c763d; margin-top: 0; font-size: 22px;">Calculation Completed</h3>

        <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">User Inputs</h4>
        <p><strong>Program:</strong> {program}</p>
        <p><strong>Households:</strong> {hh}</p>
        <p><strong>Produce per HH:</strong> {produce_per_hh}</p>
        <p><strong>Purchased per HH:</strong> {purchased_per_hh}</p>
        <p><strong>Donated per HH:</strong> {donated_per_hh}</p>
        <p><strong>Distance:</strong> {miles} miles</p>

        <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

        <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">Calculator Outputs</h4>
        <p><strong>Total Weight:</strong> {total_lbs:.2f} lbs</p>
        <p><strong>Base Food Cost:</strong> ${base_cost:.2f}</p>
        <p><strong>Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb):</strong> ${fixed_cost:.2f}</p>
        <p><strong>Transport Cost (@ $0.02/lb/mile):</strong> ${transport_cost:.2f}</p>

        <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

        <h4 style="margin: 12px 0 5px; font-size: 18px; color: #6BA539;">Food Cost Per lb Per HH</h4>
        <p><strong>Produce:</strong> {produce_per_hh} lbs × ${prod_cost:.2f} = ${prod_cost_hh:.2f} per HH</p>
        <p><strong>Purchased:</strong> {purchased_per_hh} lbs × ${purch_cost:.2f} = ${purch_cost_hh:.2f} per HH</p>
        <p><strong>Donated:</strong> {donated_per_hh} lbs × ${don_cost:.2f} = ${don_cost_hh:.2f} per HH</p>

        <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

        <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">Final Outputs</h4>
        <p><strong>Delivery Cost (Food + Transport):</strong> ${delivery_cost:.2f}</p>
        <p><strong>Total Cost:</strong> ${total_cost:.2f}</p>
        <p><strong>Blended Cost per lb:</strong> ${total_cost / total_lbs:.4f}</p>
    </div>
</div>
""")

    st.markdown(result_html, unsafe_allow_html=True)
