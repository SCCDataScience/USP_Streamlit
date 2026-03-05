import streamlit as st
import pandas as pd
import plotly.express as px
import json

# 1. Page Configuration
st.set_page_config(page_title="Wellbeing & Places 2.0", layout="wide")

# 2. Sidebar Setup
st.sidebar.title("🛠 Custom Index")
st.sidebar.markdown("Define your indicators and weighting below.")

with st.sidebar.expander("📊 Indicators", expanded=True):
    health = st.multiselect("Health", ["Life Expectancy", "Mental Health", "Air Quality"], default=["Life Expectancy"])
    econ = st.multiselect("Economy", ["GVA", "Employment Rate", "Business Start-ups"], default=["Employment Rate"])
    place = st.multiselect("Place", ["Green Space", "Crime Rate", "Connectivity"])

weight_method = st.sidebar.radio("Weighting Logic", ["Equal", "Statistical (PCA)", "Inverse-Covariance"])

# Methodology blurb
st.sidebar.markdown("---")
st.sidebar.subheader("How is it calculated?")
with st.sidebar.expander("Read about Weighting"):
    st.write("**Equal:** Every indicator contributes the same amount to the score. Best for simple comparisons.")
    st.write("**Statistical (PCA):** The tool looks for patterns in the data to see which factors are the strongest 'drivers' of wellbeing in our region.")
    st.write("**Inverse-Covariance:** This ensures we don't 'double-count' issues that are highly related (like unemployment and low income).")

# 3. Main Header & Metrics
st.title("🏙️ Strategic Places & Wellbeing Explorer")
st.markdown("---")

# KPI Row
m1, m2, m3 = st.columns(3)
m1.metric("Average Score", "72.4", "+2.1%")
m2.metric("Economic Growth", "£34.2k GVA", "+0.5%")
m3.metric("Equity Gap", "12.4%", "-1.2%")

# Change to this for better compatibility with older versions:
# col1, col2, col3 = st.columns([1, 1, 1])
# with col1:
    # st.metric("Average Score", "72.4", "2.1%")
# with col2:
    # st.metric("Economic Growth", "£34.2k", "0.5%")
# with col3:
    # st.metric("Equity Gap", "12.4%", "-1.2%")

# 4. Map and Data Section
col_map, col_table = st.columns([2, 1])

with col_map:
    st.subheader("Regional Heatmap")
    try:
        with open('boundaries.geojson') as f:
            geo = json.load(f)
        
        # Mocking data to match your GeoJSON 'NM' property
        mock_map_data = pd.DataFrame({
            "Borough": ["Elmbridge", "Woking", "Spelthorne", "Runnymede", "Guildford"], # Update these to match your GeoJSON
            "Score": [85, 72, 64, 91, 77]
        })

        fig = px.choropleth_mapbox(
            mock_map_data, geojson=geo, locations="Borough",
            featureidkey="properties.NM", color="Score",
            color_continuous_scale="Viridis", mapbox_style="open-street-map",
            zoom=10, center={"lat": 51.3, "lon": -0.4}, opacity=0.6
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Map Preview: Please check 'boundaries.geojson' is in the folder.")

with col_table:
    st.subheader("Rankings")
    st.table(pd.DataFrame({
        "Area": ["North", "South", "East", "West"],
        "Index": [85.2, 72.1, 64.9, 91.0]
    }))

# 5. Action Buttons
st.markdown("---")
if st.button("📥 Export Policy Briefing (PDF)"):
    st.success("Briefing generated!")