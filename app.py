import streamlit as st
import base64

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
    'SP': {'produce': 16, 'purchased': 5, 'donated': 2},
    'BP': {'produce': 4, 'purchased': 4, 'donated': 4},
    'MP': {'produce': 16, 'purchased': 5, 'donated': 2},
    'PP': {'produce': 24, 'purchased': 0, 'donated': 0}
}

# === UI ===
st.set_page_config(page_title="FSD Cost Estimator", layout="centered")

# === Logo ===
with open("FSD LOGO.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
st.markdown(f"<div style='text-align:center'><img src='data:image/png;base64,{encoded_image}' style='height:80px; margin-bottom:20px;'/></div>", unsafe_allow_html=True)

st.title("Delivery Cost Estimator")

# === Inputs ===
program = st.selectbox("1. Which program is this?", list(lbs_per_hh.keys()))
hh = st.number_input("2. How many households are served?", min_value=1, value=350)
prod_lb = st.number_input("3. How many lbs of produce per HH?", min_value=0.0, value=float(lbs_per_hh[program]["produce"]))
purch_lb = st.number_input("4. How many lbs of purchased per HH?", min_value=0.0, value=float(lbs_per_hh[program]["purchased"]))
don_lb = st.number_input("5. How many lbs of donated per HH?", min_value=0.0, value=float(lbs_per_hh[program]["donated"]))
miles = st.number_input("6. How many miles will this delivery travel?", min_value=0.0, value=30.0)

if st.button("Calculate & Estimate"):
    # Select cost/lb
    produce_cost = cost_per_lb_fixed["produce"]
    donated_cost = cost_per_lb_fixed["donated"]
    purchased_cost = cost_per_lb_fixed["purchased_bp"] if program == "BP" else cost_per_lb_fixed["purchased"]

    # Weight calculations
    prod_total = prod_lb * hh
    purch_total = purch_lb * hh
    don_total = don_lb * hh
    total_lbs = prod_total + purch_total + don_total

    # Cost calculations
    base_cost = prod_total * produce_cost + purch_total * purchased_cost + don_total * donated_cost
    fixed_cost = total_lbs * FIXED_COST_PER_LB
    transport_cost = total_lbs * miles * TRANSPORT_COST_PER_LB_PER_MILE
    delivery_cost = base_cost + transport_cost
    total_cost = base_cost + fixed_cost + transport_cost

    # Per-HH cost
    prod_cost_hh = prod_lb * produce_cost
    purch_cost_hh = purch_lb * purchased_cost
    don_cost_hh = don_lb * donated_cost

    # === Output HTML ===
    st.markdown(f"""
    <div style='background-color:#fff; border-radius:10px; padding:20px; text-align:left; max-width:500px; margin:auto;'>

        <h3 style="color: #3c763d; font-size: 22px;">Calculation Completed</h3>

        <h4 style="color: #6BA539;">User Inputs</h4>
        <p style='color:#000'><strong>Program:</strong> {program}</p>
        <p style='color:#000'><strong>Households:</strong> {hh}</p>
        <p style='color:#000'><strong>Produce per HH:</strong> {prod_lb}</p>
        <p style='color:#000'><strong>Purchased per HH:</strong> {purch_lb}</p>
        <p style='color:#000'><strong>Donated per HH:</strong> {don_lb}</p>
        <p style='color:#000'><strong>Distance:</strong> {miles} miles</p>

        <hr style="border-top: 1px solid #ccc;" />

        <h4 style="color: #6BA539;">Calculator Outputs</h4>
        <p style='color:#000'><strong>Total Weight:</strong> {total_lbs:.2f} lbs</p>
        <p style='color:#000'><strong>Base Food Cost:</strong> ${base_cost:.2f}</p>
        <p style='color:#000'><strong>Fixed Cost (@ ${FIXED_COST_PER_LB:.4f}/lb):</strong> ${fixed_cost:.2f}</p>
        <p style='color:#000'><strong>Transport Cost (@ $0.02/lb/mile):</strong> ${transport_cost:.2f}</p>

        <hr style="border-top: 1px solid #ccc;" />

        <h4 style="color: #6BA539;">Food Cost Per lb Per HH</h4>
        <p style='color:#000'><strong>Produce:</strong> {prod_lb} lbs × ${produce_cost:.2f} = ${prod_cost_hh:.2f} per HH</p>
        <p style='color:#000'><strong>Purchased:</strong> {purch_lb} lbs × ${purchased_cost:.2f} = ${purch_cost_hh:.2f} per HH</p>
        <p style='color:#000'><strong>Donated:</strong> {don_lb} lbs × ${donated_cost:.2f} = ${don_cost_hh:.2f} per HH</p>

        <hr style="border-top: 1px solid #ccc;" />

        <h4 style="color: #6BA539;">Final Outputs</h4>
        <p style='color:#000'><strong>Delivery Cost (Food + Transport):</strong> ${delivery_cost:.2f}</p>
        <p style='color:#000'><strong>Total Cost:</strong> ${total_cost:.2f}</p>
        <p style='color:#000'><strong>Blended Cost per lb:</strong> ${total_cost / total_lbs:.4f}</p>

    </div>
    """, unsafe_allow_html=True)
