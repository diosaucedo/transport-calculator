import streamlit as st
import pandas as pd
import numpy as np
import base64

# === Load logo ===
def get_base64_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_path = get_base64_logo("FSD LOGO.png")

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ≈ 0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02

cost_per_lb_fixed = {
    'produce': 0.17,
    'purchased': 0.85,
    'donated': 0.04,
    'purchased_bp': 1.27
}

# === Default lb per HH by program ===
lbs_per_hh = {
    'AGENCY': {'produce': 10, 'purchased': 15, 'donated': 5},
    'BP': {'produce': 12, 'purchased': 18, 'donated': 3}
}

# === UI Elements ===
st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{logo_path}' style='height:80px; margin-bottom:20px;'></div>", unsafe_allow_html=True)
st.title("FSD Delivery Cost Estimator")

program = st.selectbox("1. Which program is this?", options=list(lbs_per_hh.keys()))
hh = st.number_input("2. How many households are served?", min_value=1, value=350)
prod_lb = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=float(lbs_per_hh[program]['produce']))
purch_lb = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=float(lbs_per_hh[program]['purchased']))
don_lb = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=float(lbs_per_hh[program]['donated']))
miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

# === Calculate Button ===
if st.button("Calculate & Estimate"):
    # Use correct purchased cost
    if program == 'BP':
        purchased_cost = cost_per_lb_fixed['purchased_bp']
    else:
        purchased_cost = cost_per_lb_fixed['purchased']
    
    produce_cost = cost_per_lb_fixed['produce']
    donated_cost = cost_per_lb_fixed['donated']

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

    # === Output ===
    st.markdown("### Calculation Completed")
    st.subheader("User Inputs")
    st.markdown(f"- **Program:** {program}")
    st.markdown(f"- **Households:** {hh}")
    st.markdown(f"- **Produce per HH:** {prod_lb}")
    st.markdown(f"- **Purchased per HH:** {purch_lb}")
    st.markdown(f"- **Donated per HH:** {don_lb}")
    st.markdown(f"- **Distance:** {miles} miles")

    st.subheader("Calculator Outputs")
    st.markdown(f"- **Total Weight:** {total_lbs:.2f} lbs")
    st.markdown(f"- **Base Food Cost:** ${base_cost:.2f}")
    st.markdown(f"- **Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb):** ${fixed_cost:.2f}")
    st.markdown(f"- **Transport Cost (@ $0.02/lb/mile):** ${transport_cost:.2f}")

    st.subheader("Food Cost Per lb Per HH")
    st.markdown(f"- **Produce:** {prod_lb} lbs × ${produce_cost:.2f} = ${prod_cost_hh:.2f} per HH")
    st.markdown(f"- **Purchased:** {purch_lb} lbs × ${purchased_cost:.2f} = ${purch_cost_hh:.2f} per HH")
    st.markdown(f"- **Donated:** {don_lb} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f} per HH")

    st.subheader("Final Outputs")
    st.markdown(f"- **Delivery Cost (Food + Transport):** ${delivery_cost:.2f}")
    st.markdown(f"- **Total Cost (including fixed):** ${total_cost:.2f}")
    st.markdown(f"- **Blended Cost per lb:** ${total_cost / total_lbs:.4f}")
