import streamlit as st
import pandas as pd
import base64
import numpy as np

# === Page Config ===
st.set_page_config(layout="centered")

# === Load logo ===
def load_image():
    with open("FSD LOGO.png", "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ≈ 0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02

# === Cost/lb by group ===
cost_per_lb_fixed = {
    'produce': 0.17,
    'purchased': 0.85,
    'donated': 0.04,
    'purchased_bp': 1.27
}

# === Page Layout ===
# Center-align the logo
st.markdown(f'<div style="text-align: center;"><img src="{load_image()}" style="height: 80px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)

# Input fields
st.markdown("<h4>1. Which program is this?</h4>", unsafe_allow_html=True)
program = st.selectbox("", lbs_per_hh.keys(), key="program")

st.markdown("<h4>2. How many households are served?</h4>", unsafe_allow_html=True)
hh = st.number_input("", value=350, step=1, key="hh")

st.markdown("<h4>3. How many lbs of produce per HH?</h4>", unsafe_allow_html=True)
prod_lb = st.number_input("", value=float(lbs_per_hh[program]['produce']), key="produce")

st.markdown("<h4>4. How many lbs of purchased per HH?</h4>", unsafe_allow_html=True)
purch_lb = st.number_input("", value=float(lbs_per_hh[program]['purchased']), key="purchased")

st.markdown("<h4>5. How many lbs of donated per HH?</h4>", unsafe_allow_html=True)
don_lb = st.number_input("", value=float(lbs_per_hh[program]['donated']), key="donated")

st.markdown("<h4>6. How many miles will this delivery travel?</h4>", unsafe_allow_html=True)
miles = st.number_input("", value=30.0, key="miles")

# Calculate button
if st.button("Calculate & Estimate", type="primary"):
    # Cost calculations
    produce_cost = cost_per_lb_fixed['produce']
    donated_cost = cost_per_lb_fixed['donated']
    purchased_cost = cost_per_lb_fixed['purchased_bp'] if program == 'BP' else cost_per_lb_fixed['purchased']

    prod_total = prod_lb * hh
    purch_total = purch_lb * hh
    don_total = don_lb * hh
    total_lbs = prod_total + purch_total + don_total

    base_cost = (
        prod_total * produce_cost +
        purch_total * purchased_cost +
        don_total * donated_cost
    )
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    prod_cost_hh = prod_lb * produce_cost
    purch_cost_hh = purch_lb * purchased_cost
    don_cost_hh = don_lb * donated_cost

    # Output display
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; display: inline-block; text-align: left; max-width: 360px;">
            <h3 style="color: #3c763d; margin-top: 0; font-size: 22px;">Calculation Completed</h3>

            <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">User Inputs</h4>
            <p style='color:#000;'><strong>Program:</strong> {program}</p>
            <p style='color:#000;'><strong>Households:</strong> {hh}</p>
            <p style='color:#000;'><strong>Produce per HH:</strong> {prod_lb}</p>
            <p style='color:#000;'><strong>Purchased per HH:</strong> {purch_lb}</p>
            <p style='color:#000;'><strong>Donated per HH:</strong> {don_lb}</p>
            <p style='color:#000;'><strong>Distance:</strong> {miles} miles</p>

            <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

            <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">Calculator Outputs</h4>
            <p style='color:#000;'><strong>Total Weight:</strong> {total_lbs:.2f} lbs</p>
            <p style='color:#000;'><strong>Base Food Cost:</strong> ${base_cost:.2f}</p>
            <p style='color:#000;'><strong>Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb):</strong> ${fixed_cost:.2f}</p>
            <p style='color:#000;'><strong>Transport Cost (@ $0.02/lb/mile):</strong> ${transport_cost:.2f}</p>

            <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

            <h4 style="margin: 12px 0 5px; font-size: 18px; color: #6BA539;">Food Cost Per lb Per HH</h4>
            <p style='color:#000;'><strong>Produce:</strong> {prod_lb} lbs × ${produce_cost:.2f} = ${prod_cost_hh:.2f} per HH</p>
            <p style='color:#000;'><strong>Purchased:</strong> {purch_lb} lbs × ${purchased_cost:.2f} = ${purch_cost_hh:.2f} per HH</p>
            <p style='color:#000;'><strong>Donated:</strong> {don_lb} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f} per HH</p>

            <hr style="border: none; border-top: 1px solid #ccc; margin: 12px 0;">

            <h4 style="margin-bottom: 5px; font-size: 18px; color: #6BA539;">Final Outputs</h4>
            <p style='color:#000;'><strong>Delivery Cost (Food + Transport):</strong> ${delivery_cost:.2f}</p>
            <p style='color:#000;'><strong>Total Cost:</strong> ${total_cost:.2f}</p>
            <p style='color:#000;'><strong>Blended Cost per lb:</strong> ${total_cost / total_lbs:.4f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
