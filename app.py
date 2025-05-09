import streamlit as st
import pandas as pd
import numpy as np

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ≈ 0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02

cost_per_lb_fixed = {
    'produce': 0.17,
    'purchased': 0.85,
    'donated': 0.04,
    'purchased_bp': 1.27
}

lbs_per_hh = {
    'AGENCY': {'produce': 16, 'purchased': 0, 'donated': 0},
    'MEM': {'produce': 16, 'purchased': 3, 'donated': 4},
    'BP': {'produce': 4, 'purchased': 4, 'donated': 4},
    'MP': {'produce': 16, 'purchased': 5, 'donated': 2},
    'PP': {'produce': 24, 'purchased': 0, 'donated': 0},
    'SP': {'produce': 16, 'purchased': 5, 'donated': 2}
}

# === UI ===
st.set_page_config(page_title="FSD Delivery Cost Estimator", layout="centered")
st.image("FSD LOGO.png", width=160)
st.title("Delivery Cost Estimator")

program = st.selectbox("1. Which program is this?", list(lbs_per_hh.keys()))
hh = st.number_input("2. How many households are served?", min_value=1, value=350)
prod_lb = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=float(lbs_per_hh[program]['produce']))
purch_lb = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=float(lbs_per_hh[program]['purchased']))
don_lb = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=float(lbs_per_hh[program]['donated']))
miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

if st.button("Calculate & Estimate"):
    # === Use fixed cost per lb ===
    produce_cost = cost_per_lb_fixed['produce']
    donated_cost = cost_per_lb_fixed['donated']
    purchased_cost = cost_per_lb_fixed['purchased_bp'] if program == 'BP' else cost_per_lb_fixed['purchased']

    prod_total = prod_lb * hh
    purch_total = purch_lb * hh
    don_total = don_lb * hh
    total_lbs = prod_total + purch_total + don_total

    base_cost = prod_total * produce_cost + purch_total * purchased_cost + don_total * donated_cost
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    prod_cost_hh = prod_lb * produce_cost
    purch_cost_hh = purch_lb * purchased_cost
    don_cost_hh = don_lb * donated_cost

    st.markdown(f"""
    <div style='background-color: #fff; border-radius: 10px; padding: 20px; max-width: 600px; margin: auto; text-align: left;'>
        <h3 style='color: #3c763d;'>Calculation Completed</h3>

        <h4 style='color: #6BA539;'>User Inputs</h4>
        <p style='color:#000;'>Program: {program}</p>
        <p style='color:#000;'>Households: {hh}</p>
        <p style='color:#000;'>Produce per HH: {prod_lb}</p>
        <p style='color:#000;'>Purchased per HH: {purch_lb}</p>
        <p style='color:#000;'>Donated per HH: {don_lb}</p>
        <p style='color:#000;'>Distance: {miles} miles</p>

        <hr style="border-top: 1px solid #ccc;">

        <h4 style='color: #6BA539;'>Calculator Outputs</h4>
        <p style='color:#000;'>Total Weight: {total_lbs:.2f} lbs</p>
        <p style='color:#000;'>Base Food Cost: ${base_cost:.2f}</p>
        <p style='color:#000;'>Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb): ${fixed_cost:.2f}</p>
        <p style='color:#000;'>Transport Cost (@ $0.02/lb/mile): ${transport_cost:.2f}</p>

        <hr style="border-top: 1px solid #ccc;">

        <h4 style='color: #6BA539;'>Food Cost Per lb Per HH</h4>
        <p style='color:#000;'>Produce: {prod_lb} lbs × ${produce_cost:.2f} = ${prod_cost_hh:.2f} per HH</p>
        <p style='color:#000;'>Purchased: {purch_lb} lbs × ${purchased_cost:.2f} = ${purch_cost_hh:.2f} per HH</p>
        <p style='color:#000;'>Donated: {don_lb} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f} per HH</p>

        <hr style="border-top: 1px solid #ccc;">

        <h4 style='color: #6BA539;'>Final Outputs</h4>
        <p style='color:#000;'>Delivery Cost (Food + Transport): ${delivery_cost:.2f}</p>
        <p style='color:#000; font-weight: bold;'>Total Cost: ${total_cost:.2f}</p>
        <p style='color:#000; font-weight: bold;'>Blended Cost per lb: ${total_cost / total_lbs:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
