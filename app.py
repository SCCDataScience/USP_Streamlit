import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Understand Surrey's 2.0", layout="wide")

# --- DATA LOADING & NORMALIZATION ---
@st.cache_data
def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    
    # Normalization Logic (The Math)
    normalized_list = []
    for indicator in df['Indicator_Name'].unique():
        subset = df[df['Indicator_Name'] == indicator].copy()
        
        v_min = subset['Value'].min()
        v_max = subset['Value'].max()
        direction = subset['Direction'].iloc[0]
        
        # Min-Max Scaling (Handling cases where min == max to avoid division by zero)
        if v_max - v_min != 0:
            subset['Norm_Value'] = (subset['Value'] - v_min) / (v_max - v_min)
        else:
            subset['Norm_Value'] = 0.5
            
        # Flip if Direction is Negative (e.g., higher crime = lower wellbeing)
        if direction == 'Negative':
            subset['Norm_Value'] = 1 - subset['Norm_Value']
            
        normalized_list.append(subset)
    
    return pd.concat(normalized_list)

# Load your new test data
try:
    df = load_and_normalize('USP_test.csv')
except FileNotFoundError:
    st.error("⚠️ 'USP_test.csv' not found. Please upload it to your GitHub repo.")
    st.stop()

# 2. Sidebar - User Choices
st.sidebar.title("🎮 Tool Controls")
mode = st.sidebar.radio("Analysis Mode", ["Single Indicator Performance", "Bespoke Index Builder"])

if mode == "Single Indicator Performance":
    selected_ind = st.sidebar.selectbox("Select Indicator", df['Indicator_Name'].unique())
    display_data = df[df['Indicator_Name'] == selected_ind]
    chart_title = f"Trend: {selected_ind}"
    map_color_col = "Value" # Show raw values for single indicator
    unit_label = display_data['Unit'].iloc[0]
else:
    st.sidebar.markdown("Select indicators to combine into your index:")
    selected_inds = st.sidebar.multiselect("Indicators", df['Indicator_Name'].unique(), default=df['Indicator_Name'].unique()[:3])
    
    # Calculate Index: Average the Normalized Values across selected indicators
    display_data = df[df['Indicator_Name'].isin(selected_inds)].groupby(['Area_Name', 'Year', 'Area_Code']).agg({'Norm_Value': 'mean'}).reset_index()
    chart_title = "Trend: Bespoke Index Score (0-1 Scale)"
    map_color_col = "Norm_Value"
    unit_label = "Index Score"

# 3. Main Header
st.title("🏙️ Understanding Surrey's Places")
st.markdown(f"**Mode:** {mode}")

# 4. Trends Over Time (The Line Chart)
st.subheader(chart_title)
line_fig = px.line(display_data, x="Year", y=map_color_col, color="Area_Name", markers=True,
                  labels={map_color_col: unit_label})
line_fig.update_layout(xaxis_type='category') # Keeps years as discrete steps
st.plotly_chart(line_fig, use_container_width=True)

st.divider()

# 5. Map and Rankings (Latest Year Only)
latest_year = display_data['Year'].max()
map_data = display_data[display_data['Year'] == latest_year]

col_map, col_table = st.columns([2, 1])

with col_map:
    st.subheader(f"Regional Snapshot ({latest_year})")
    try:
        with open('boundaries.geojson') as f:
            geo = json.load(f)
        
        fig = px.choropleth_mapbox(
            map_data, geojson=geo, locations="Area_Name",
            featureidkey="properties.NM", color=map_color_col,
            color_continuous_scale="Viridis", mapbox_style="open-street-map",
            zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.6
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("⚠️ Map error: Ensure 'boundaries.geojson' is present and names match.")

with col_table:
    st.subheader("Current Rankings")
    table_df = map_data[['Area_Name', map_color_col]].sort_values(by=map_color_col, ascending=False)
    st.dataframe(table_df, hide_index=True, use_container_width=True)
