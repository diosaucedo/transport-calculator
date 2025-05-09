# app.py

import streamlit as st
import pandas as pd
import numpy as np
import base64

# === Data for default values ===
lbs_per_hh = {
    'AGENCY': {'produce': 7.5, 'purchased': 2.2, 'donated': 3.3},
    'BP': {'produce': 10.0, 'purchased': 5.5, 'donated': 1.0}
}

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ≈ 0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02
cost_per_lb_fixed = {
    'produce': 0.17,
    'purchased': 0.85,
    'donated': 0.04,
    'purchased_bp': 1.27
}

# === Set page config ===
st.set_page_config(page_title="Delivery Cost Estimator", layout="centered")

# === Load logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_path = f"data:image/png;base64,{encoded_image}"

st.markdown(f"""
<div style='text-align: center;'>
    <img src="{logo_path}" style='height: 80px; margin-bottom: 20px;'/>
</div>
""", unsafe_allow_html=True)

# === Form ===
st.markdown("### 1. Which program is this?")
program = st.selectbox("", options=list(lbs_per_hh.keys()))

st.markdown("### 2. How many households are served?")
hh = st.number_input("", min_value=1, value=350, step=1)

st.markdown("### 3. How many lbs of produce per HH?")
produce = st.number_input("", min_value=0.0, value=lbs_per_hh[program]['produce'], step=0.1)

st.markdown("### 4. How many lbs of purchased per HH?")
purchased = st.number_input("", min_value=0.0, value=lbs_per_hh[program]['purchased'], step=0.1)

st.markdown("### 5. How many lbs of donated per HH?")
donated = st.number_input("", min_value=0.0, value=lbs_per_hh[program]['donated'], step=0.1)

st.markdown("### 6. How many miles will this delivery travel?")
miles = st.number_input("", min_value=0.0, value=30.0, step=0.5)

# === Calculate button ===
if st.button("Calculate & Estimate"):
    # Per-lb costs
    produce_cost = cost_per_lb_fixed['produce']
    donated_cost = cost_per_lb_fixed['donated']
    purchased_cost = cost_per_lb_fixed['purchased_bp'] if program == 'BP' else cost_per_lb_fixed['purchased']

    # Weight totals
    prod_total = produce * hh
    purch_total = purchased * hh
    don_total = donated * hh
    total_lbs = prod_total + purch_total + don_total

    # Costs
    base_cost = prod_total * produce_cost + purch_total * purchased_cost + don_total * donated_cost
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    prod_cost_hh = produce * produce_cost
    purch_cost_hh = purchased * purchased_cost
    don_cost_hh = donated * donated_cost

    # === Output display ===
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; display: inline-block; text-align: left; max-width: 360px;">
            <h3 style="color: #3c763d; margin-top: 0; font-size: 22px;">Calculation Completed</h3>

            <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">User Inputs</h4>
            <p style='color:#000;'><strong>Program:</strong> {program}</p>
            <p style='color:#000;'><strong>Households:</strong> {hh}</p>
            <p style='color:#000;'><strong>Produce per HH:</strong> {produce}</p>
            <p style='color:#000;'><strong>Purchased per HH:</strong> {purchased}</p>
            <p style='color:#000;'><strong>Donated per HH:</strong> {donated}</p>
            <p style='color:#000;'><strong>Distance:</strong> {miles} miles</p>

            <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

            <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">Calculator Outputs</h4>
            <p style='color:#000;'><strong>Total Weight:</strong> {total_lbs:.2f} lbs</p>
            <p style='color:#000;'><strong>Base Food Cost:</strong> ${base_cost:.2f}</p>
            <p style='color:#000;'><strong>Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb):</strong> ${fixed_cost:.2f}</p>
            <p style='color:#000;'><strong>Transport Cost (@ $0.02/lb/mile):</strong> ${transport_cost:.2f}</p>

            <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

            <h4 style="margin: 12px 0 5px; font-size: 18px; color: #6BA539;">Food Cost Per lb Per HH</h4>
            <p style='color:#000;'><strong>Produce:</strong> {produce} lbs × ${produce_cost:.2f} = ${prod_cost_hh:.2f} per HH</p>
            <p style='color:#000;'><strong>Purchased:</strong> {purchased} lbs × ${purchased_cost:.2f} = ${purch_cost_hh:.2f} per HH</p>
            <p style='color:#000;'><strong>Donated:</strong> {donated} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f} per HH</p>

            <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

            <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">Final Outputs</h4>
            <p style='color:#000;'><strong>Delivery Cost (Food + Transport):</strong> ${delivery_cost:.2f}</p>
            <p style='color:#000;'><strong>Total Cost:</strong> ${total_cost:.2f}</p>
            <p style='color:#000;'><strong>Blended Cost per lb:</strong> ${total_cost / total_lbs:.4f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
