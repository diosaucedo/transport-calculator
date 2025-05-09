import streamlit as st
import pandas as pd
import base64

# === Load logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
logo_path = f"data:image/png;base64,{encoded_image}"

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ≈ 0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02

cost_per_lb_fixed = {
    'produce': 0.17,
    'purchased': 0.85,
    'donated': 0.04,
    'purchased_bp': 1.27
}

# === Default lbs per HH by program (placeholder dictionary, define yours) ===
lbs_per_hh = {
    'AGENCY': {'produce': 3.5, 'purchased': 1.5, 'donated': 2.0},
    'BP': {'produce': 2.0, 'purchased': 4.0, 'donated': 1.0}
}

# === Layout ===
st.markdown(f"<div style='text-align: center;'><img src='{logo_path}' style='height:80px;'></div>", unsafe_allow_html=True)
st.header("FSD Cost Calculator")

# === Inputs ===
program = st.selectbox("1. Which program is this?", options=list(lbs_per_hh.keys()))
hh = st.number_input("2. How many households are served?", min_value=1, value=350)
prod_lb = st.number_input("3. How many lbs of produce per HH?", value=lbs_per_hh[program]['produce'])
purch_lb = st.number_input("4. How many lbs of purchased per HH?", value=lbs_per_hh[program]['purchased'])
don_lb = st.number_input("5. How many lbs of donated per HH?", value=lbs_per_hh[program]['donated'])
miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

if st.button("Calculate & Estimate"):
    # === Cost logic ===
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

    # === Display results ===
    st.markdown("### Calculation Completed")
    st.markdown(f"""
    **User Inputs**  
    - Program: {program}  
    - Households: {hh}  
    - Produce per HH: {prod_lb} lbs  
    - Purchased per HH: {purch_lb} lbs  
    - Donated per HH: {don_lb} lbs  
    - Distance: {miles} miles  

    **Calculator Outputs**  
    - Total Weight: {total_lbs:.2f} lbs  
    - Base Food Cost: ${base_cost:.2f}  
    - Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb): ${fixed_cost:.2f}  
    - Transport Cost (@ $0.02/lb/mile): ${transport_cost:.2f}  

    **Food Cost Per lb Per HH**  
    - Produce: {prod_lb} lbs × ${produce_cost:.2f} = ${prod_cost_hh:.2f}  
    - Purchased: {purch_lb} lbs × ${purchased_cost:.2f} = ${purch_cost_hh:.2f}  
    - Donated: {don_lb} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f}  

    **Final Outputs**  
    - Delivery Cost (Food + Transport): ${delivery_cost:.2f}  
    - Total Cost: ${total_cost:.2f}  
    - Blended Cost per lb: ${total_cost / total_lbs:.4f}  
    """)
