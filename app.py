import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Understanding Surrey's Places 2.0", layout="wide")

# --- CORE MATH FUNCTIONS ---

def normalize_series(series, direction):
    """Normalize a series to 0-1 based on its own min/max."""
    v_min, v_max = series.min(), series.max()
    if v_max == v_min:
        return series * 0 + 0.5
    norm = (series - v_min) / (v_max - v_min)
    return 1 - norm if direction == 'Negative' else norm

def apply_pca_weights(df_pivot):
    """Calculate weights using the first principal component."""
    try:
        pca = PCA(n_components=1)
        pca.fit(df_pivot)
        weights = np.abs(pca.components_[0])
        return weights / weights.sum()
    except:
        return np.array([1/df_pivot.shape[1]] * df_pivot.shape[1])

# --- DATA LOADING ---
@st.cache_data
def load_raw_data(file_path):
    return pd.read_csv(file_path)

try:
    df_raw = load_raw_data('USP_test.csv')
except FileNotFoundError:
    st.error("⚠️ 'USP_test.csv' not found.")
    st.stop()

# 2. Sidebar - Geography & Scope
st.sidebar.title("🛠️ Tool Configuration")

with st.sidebar.expander("🌍 Geography & Scope", expanded=True):
    all_areas = sorted(df_raw['Area_Name'].unique())
    selected_areas = st.multiselect("Select Areas to Display", all_areas, default=all_areas)
    
    calc_scope = st.radio(
        "Normalization Scope",
        ["Regional (Compare against all Surrey)", "Local (Compare only against selected)"],
        help="Regional: 1.0 is the best in the full dataset. Local: 1.0 is the best among only selected areas."
    )

# 3. Sidebar - Indicators & Mode
mode = st.sidebar.radio("Analysis Mode", ["Single Indicator", "Bespoke Index"])

theme_weights = {}
weight_method = "Equal Weighting"

if mode == "Single Indicator":
    selected_ind = st.sidebar.selectbox("Select Indicator", df_raw['Indicator_Name'].unique())
    indicators_to_use = [selected_ind]
else:
    with st.sidebar.expander("📊 Index Components", expanded=True):
        selected_inds = st.multiselect(
            "Select Indicators", 
            df_raw['Indicator_Name'].unique(), 
            default=df_raw['Indicator_Name'].unique()[:4]
        )
    indicators_to_use = selected_inds

    weight_method = st.sidebar.selectbox(
        "Indicator Weighting Logic", 
        ["Equal Weighting", "Statistical (PCA)"]
    )

    active_themes = df_raw[df_raw['Indicator_Name'].isin(selected_inds)]['Theme'].unique()
    if len(active_themes) > 1:
        with st.sidebar.expander("🎨 Theme Importance", expanded=True):
            for theme in active_themes:
                theme_weights[theme] = st.slider(f"{theme} Weight", 0, 100, 50)
    else:
        for theme in active_themes: theme_weights[theme] = 100

# --- DATA PROCESSING ENGINE ---

# Step 1: Filter Scope for Math
if calc_scope == "Local (Compare only against selected)":
    df_calc = df_raw[df_raw['Area_Name'].isin(selected_areas)].copy()
else:
    df_calc = df_raw.copy()

# Step 2: Normalize
normalized_list = []
for ind in indicators_to_use:
    subset = df_calc[df_calc['Indicator_Name'] == ind].copy()
    if not subset.empty:
        subset['Norm_Value'] = normalize_series(subset['Value'], subset['Direction'].iloc[0])
        normalized_list.append(subset)

if not normalized_list:
    st.warning("Please select at least one indicator.")
    st.stop()

df_norm = pd.concat(normalized_list)

# Step 3: Index Calculation
if mode == "Bespoke Index":
    theme_scores = []
    for theme in active_themes:
        theme_subset = df_norm[df_norm['Theme'] == theme]
        if weight_method == "Statistical (PCA)":
            pivot = theme_subset.pivot_table(index=['Area_Name', 'Year'], columns='Indicator_Name', values='Norm_Value').dropna()
            if not pivot.empty:
                w = apply_pca_weights(pivot)
                pivot['Theme_Score'] = (pivot * w).sum(axis=1)
                t_score = pivot[['Theme_Score']].reset_index()
            else:
                t_score = theme_subset.groupby(['Area_Name', 'Year'])['Norm_Value'].mean().reset_index().rename(columns={'Norm_Value': 'Theme_Score'})
        else:
            t_score = theme_subset.groupby(['Area_Name', 'Year'])['Norm_Value'].mean().reset_index().rename(columns={'Norm_Value': 'Theme_Score'})
        
        t_score['Theme'] = theme
        theme_scores.append(t_score)
    
    df_themes = pd.concat(theme_scores)
    total_w = sum(theme_weights.values())
    df_themes['Weighted_Score'] = df_themes.apply(lambda x: x['Theme_Score'] * (theme_weights[x['Theme']] / total_w), axis=1)
    final_data = df_themes.groupby(['Area_Name', 'Year'])['Weighted_Score'].sum().reset_index().rename(columns={'Weighted_Score': 'Final_Value'})
    unit_label = "Index Score (0-1)"
else:
    final_data = df_norm.rename(columns={'Value': 'Final_Value'})
    unit_label = df_norm['Unit'].iloc[0]

# --- VISUALIZATION ---
display_data = final_data[final_data['Area_Name'].isin(selected_areas)]
latest_year = display_data['Year'].max()
map_df = display_data[display_data['Year'] == latest_year].sort_values('Final_Value', ascending=False)

st.title("🏙️ Understanding Surrey's Places")
st.subheader("Performance Trend")
fig_line = px.line(display_data, x="Year", y="Final_Value", color="Area_Name", markers=True, labels={"Final_Value": unit_label})
fig_line.update_layout(xaxis_type='category')
st.plotly_chart(fig_line, use_container_width=True)

col_map, col_table = st.columns([2, 1])
with col_map:
    st.subheader(f"Map View ({latest_year})")
    try:
        with open('boundaries.geojson') as f: geo = json.load(f)
        fig_map = px.choropleth_mapbox(
            map_df, geojson=geo, locations="Area_Name", featureidkey="properties.NM",
            color="Final_Value", color_continuous_scale="Viridis", mapbox_style="open-street-map",
            zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.6
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    except: st.info("Map unavailable.")

with col_table:
    st.subheader("Rankings")
    st.dataframe(map_df[['Area_Name', 'Final_Value']], hide_index=True)

# --- REPORT GENERATOR ---
st.sidebar.divider()
st.sidebar.subheader("📥 Export Evidence")

report_text = f"""Understanding Surrey's Places Indes Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
--------------------------------------------------
ANALYSIS CONFIGURATION
Mode: {mode}
Normalization Scope: {calc_scope}
Indicators Used: {', '.join(indicators_to_use)}
Weighting Logic: {weight_method}
Theme Weights: {theme_weights if mode == 'Bespoke Index' else 'N/A'}

TOP PERFORMING AREAS ({latest_year})
1. {map_df.iloc[0]['Area_Name']} (Score: {map_df.iloc[0]['Final_Value']:.2f})
2. {map_df.iloc[1]['Area_Name'] if len(map_df)>1 else 'N/A'}
3. {map_df.iloc[2]['Area_Name'] if len(map_df)>2 else 'N/A'}

Bottom Performing Area: {map_df.iloc[-1]['Area_Name']}
--------------------------------------------------
Data source: USP_test.csv
"""

st.sidebar.download_button(
    label="Download Briefing Note (.txt)",
    data=report_text,
    file_name=f"Wellbeing_Report_{latest_year}.txt",
    mime="text/plain"
)
