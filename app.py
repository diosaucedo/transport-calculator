import streamlit as st
import pandas as pd

# --- Constants ---
FIXED_COST = 2607

# --- Logo ---
st.image("FSD LOGO.png", width=200)

# --- Title ---
st.markdown("## Transport Cost Calculator")
st.caption("Built for Feeding San Diego")

# --- Inputs ---
weight = st.number_input("1. What is the total weight in pounds?", min_value=0.0, step=1.0)

region_options = {
    'North Coastal (27.7 mi)': 27.7,
    'North Inland (46.6 mi)': 46.6,
    'Central (19.2 mi)': 19.2,
    'East (60.5 mi)': 60.5,
    'South (28.3 mi)': 28.3
}
region_label = st.selectbox("2. Where is the delivery going?", list(region_options.keys()))
distance = region_options[region_label]

agency = st.selectbox("3. Which agency is receiving the delivery?", [
    "Backpack - ARPA", "Backpack", "Community Partnership Events", "Emergency Food",
    "FSD Marketplace", "Mobile Pantry", "Produce Pantry", "School Pantry - ARPA",
    "School Pantry", "Temp. Summer Distributions", "Together Tour"
])

# --- Button ---
if st.button("Calculate"):
    if weight <= 0:
        st.warning("⚠️ Please enter a weight greater than 0.")
    else:
        cost_per_lb_mile = FIXED_COST / (weight * distance)

        st.success("✅ Calculation Completed")

        st.markdown(f"""
        <div style="font-size:16px; line-height:1.6;">
        <b>Agency:</b> {agency}  <br>
        <b>Region:</b> {region_label}  <br>
        <b>Weight:</b> {weight} lbs  <br>
        <b>Distance:</b> {distance} miles  <br>
        <hr>
        <b style="color:#F7941D;">Cost per lb per mile: ${cost_per_lb_mile:.2f}</b>
        </div>
        """, unsafe_allow_html=True)
