import streamlit as st
import pandas as pd
import numpy as np
import base64

# === Constants ===
FIXED_COST_PER_LB = 13044792 / 17562606  # ~0.7427
TRANSPORT_COST_PER_LB_PER_MILE = 0.02
DONATED_COST_PER_LB = 0.04

# === Program Cost Estimates (derived from model) ===
cost_estimates = pd.DataFrame([
    {'PROGRAM': 'AGENCY', 'estimated_produce_cost_per_lb': 0.079, 'estimated_purchased_cost_per_lb': 0.85},
    {'PROGRAM': 'BP', 'estimated_produce_cost_per_lb': 0.17, 'estimated_purchased_cost_per_lb': 1.27},
    {'PROGRAM': 'MEM', 'estimated_produce_cost_per_lb': 0.12, 'estimated_purchased_cost_per_lb': 0.91},
    {'PROGRAM': 'MP', 'estimated_produce_cost_per_lb': 0.11, 'estimated_purchased_cost_per_lb': 0.89},
    {'PROGRAM': 'PP', 'estimated_produce_cost_per_lb': 0.10, 'estimated_purchased_cost_per_lb': 0.93},
    {'PROGRAM': 'SP', 'estimated_produce_cost_per_lb': 0.13, 'estimated_purchased_cost_per_lb': 0.96}
])

# === Default lbs per HH ===
lbs_per_hh = {
    'AGENCY': {'produce': 4, 'purchased': 6, 'donated': 5},
    'MEM': {'produce': 3, 'purchased': 5, 'donated': 2},
    'BP': {'produce': 2, 'purchased': 7, 'donated': 1},
    'MP': {'produce': 3, 'purchased': 4, 'donated': 2},
    'PP': {'produce': 3, 'purchased': 6, 'donated': 2},
    'SP': {'produce': 5, 'purchased': 5, 'donated': 1}
}

# === UI ===
st.set_page_config(page_title="Delivery Cost Calculator", layout="centered")
st.markdown("<h2 style='text-align: center;'>Delivery Cost Calculator</h2>", unsafe_allow_html=True)

# Logo
with open("FSD LOGO.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()
st.markdown(f"<div style='text-align:center'><img src='data:image/png;base64,{encoded}' style='height:80px; margin-bottom:20px;'/></div>", unsafe_allow_html=True)

# Form Inputs
program = st.selectbox("1. Which program is this?", options=list(lbs_per_hh.keys()))
hh = st.number_input("2. How many households are served?", min_value=0, value=0, step=1)
prod_lb = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=0.0)
purch_lb = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=0.0)
don_lb = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=0.0)
miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=0.0)

# Calculate
if st.button("Calculate & Estimate"):
    row = cost_estimates[cost_estimates['PROGRAM'] == program].iloc[0]
    produce_cost = row['estimated_produce_cost_per_lb']
    purchased_cost = row['estimated_purchased_cost_per_lb']

    prod_total = prod_lb * hh
    purch_total = purch_lb * hh
    don_total = don_lb * hh
    total_lbs = prod_total + purch_total + don_total

    base_cost = prod_total * produce_cost + purch_total * purchased_cost + don_total * DONATED_COST_PER_LB
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    prod_cost_hh = prod_lb * produce_cost
    purch_cost_hh = purch_lb * purchased_cost
    don_cost_hh = don_lb * DONATED_COST_PER_LB

    # Display results
    st.markdown("""
    <div style='background-color: #fff; border-radius: 10px; padding: 20px; max-width: 480px; margin:auto;'>
        <h4 style='color: #6BA539;'>User Inputs</h4>
        <p><strong>Program:</strong> {}</p>
        <p><strong>Households:</strong> {}</p>
        <p><strong>Produce per HH:</strong> {}</p>
        <p><strong>Purchased per HH:</strong> {}</p>
        <p><strong>Donated per HH:</strong> {}</p>
        <p><strong>Distance:</strong> {} miles</p>
        <hr>
        <h4 style='color: #6BA539;'>Calculator Outputs</h4>
        <p><strong>Total Weight:</strong> {:.2f} lbs</p>
        <p><strong>Base Food Cost:</strong> ${:.2f}</p>
        <p><strong>Fixed Cost (@ ${:.4f}/lb):</strong> ${:.2f}</p>
        <p><strong>Transport Cost (@ $0.02/lb/mile):</strong> ${:.2f}</p>
        <hr>
        <h4 style='color: #6BA539;'>Food Cost Per lb Per HH</h4>
        <p><strong>Produce:</strong> {:.1f} lbs × ${:.3f} = ${:.2f} per HH</p>
        <p><strong>Purchased:</strong> {:.1f} lbs × ${:.3f} = ${:.2f} per HH</p>
        <p><strong>Donated:</strong> {:.1f} lbs × ${:.2f} = ${:.2f} per HH</p>
        <hr>
        <h4 style='color: #6BA539;'>Final Outputs</h4>
        <p><strong>Delivery Cost (Food + Transport):</strong> ${:.2f}</p>
        <p><strong>Total Cost:</strong> ${:.2f}</p>
        <p><strong>Blended Cost per lb:</strong> ${:.4f}</p>
    </div>
    """.format(
        program, hh, prod_lb, purch_lb, don_lb, miles,
        total_lbs, base_cost, FIXED_COST_PER_LB, fixed_cost, transport_cost,
        prod_lb, produce_cost, prod_cost_hh,
        purch_lb, purchased_cost, purch_cost_hh,
        don_lb, DONATED_COST_PER_LB, don_cost_hh,
        delivery_cost, total_cost, total_cost / total_lbs if total_lbs else 0
    ), unsafe_allow_html=True)
```
