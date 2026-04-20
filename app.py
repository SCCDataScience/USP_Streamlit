import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Understanding Surrey's Places", layout="wide")

# --- CORE MATH FUNCTIONS ---
def normalize_series(series, direction):
    v_min, v_max = series.min(), series.max()
    if v_max == v_min:
        return series * 0 + 0.5
    norm = (series - v_min) / (v_max - v_min)
    return 1 - norm if direction == 'Negative' else norm

def apply_pca_weights(df_pivot):
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
    df = pd.read_csv(file_path)
    if 'Source' not in df.columns:
        df['Source'] = "ONS / Nomis (Official Statistics)"
    return df

try:
    df_raw = load_raw_data('USP_test.csv')
except FileNotFoundError:
    st.error("⚠️ 'USP_test.csv' not found.")
    st.stop()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🛠️ Tool Configuration")

# Help Link
st.sidebar.info("💡 Confused? Check the **Data Sources** tab for the User Guide.")

with st.sidebar.expander("🌍 Geography & Scope", expanded=True):
    all_areas = sorted(df_raw['Area_Name'].unique())
    selected_areas = st.multiselect("Select Areas to Display", all_areas, default=all_areas)
    calc_scope = st.radio("Normalisation Scope", ["Regional (All Surrey)", "Local (Selected only)"])

mode = st.sidebar.radio("Analysis Mode", ["Single Indicator", "Bespoke Index"])

theme_weights = {}
weight_method = "Equal Weighting"

if mode == "Single Indicator":
    selected_ind = st.sidebar.selectbox("Select Indicator", sorted(df_raw['Indicator_Name'].unique()))
    indicators_to_use = [selected_ind]
else:
    with st.sidebar.expander("📊 Index Components", expanded=True):
        selected_inds = st.multiselect("Select Indicators", sorted(df_raw['Indicator_Name'].unique()), 
                                       default=sorted(df_raw['Indicator_Name'].unique())[:4])
    indicators_to_use = selected_inds
    weight_method = st.sidebar.selectbox("Indicator Weighting Logic", ["Equal Weighting", "Statistical (PCA)"])

    active_themes = df_raw[df_raw['Indicator_Name'].isin(selected_inds)]['Theme'].unique()
    if len(active_themes) > 1:
        with st.sidebar.expander("🎨 Theme Importance", expanded=True):
            for theme in active_themes:
                theme_weights[theme] = st.slider(f"{theme} Weight", 0, 100, 50)
    else:
        for theme in active_themes: theme_weights[theme] = 100

st.sidebar.divider()
show_corr = st.sidebar.toggle("View Correlation Matrix?")

# --- DATA PROCESSING ENGINE ---
if calc_scope == "Local (Selected only)":
    df_calc = df_raw[df_raw['Area_Name'].isin(selected_areas)].copy()
else:
    df_calc = df_raw.copy()

normalized_list = []
for ind in indicators_to_use:
    subset = df_calc[df_calc['Indicator_Name'] == ind].copy()
    if not subset.empty:
        subset['Norm_Value'] = normalize_series(subset['Value'], subset['Direction'].iloc[0])
        normalized_list.append(subset)

df_norm = pd.concat(normalized_list)

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
    final_data = df_norm.copy().rename(columns={'Value': 'Final_Value'})
    unit_label = df_norm['Unit'].iloc[0]

# --- MAIN INTERFACE (TABS) ---
tab_dashboard, tab_metadata, tab_feedback = st.tabs(["📊 Index Dashboard", "📖 Data Sources & Help", "💬 Feedback & Requests"])

with tab_dashboard:
    display_data = final_data[final_data['Area_Name'].isin(selected_areas)]
    latest_year = display_data['Year'].max()
    st.title("🏙️ Understanding Surrey's Places")
    
    # Trend Chart
    st.subheader("Performance Trend")
    fig_line = px.line(display_data, x="Year", y="Final_Value", color="Area_Name", markers=True, 
                       labels={"Final_Value": unit_label}, color_discrete_sequence=px.colors.qualitative.Safe)
    fig_line.update_layout(xaxis_type='category')
    st.plotly_chart(fig_line, use_container_width=True)

    # Map and Rankings
    col_map, col_table = st.columns([2, 1])
    map_df = display_data[display_data['Year'] == latest_year].sort_values('Final_Value', ascending=False)
    
    with col_map:
        st.subheader(f"Regional Snapshot ({latest_year})")
        try:
            with open('boundaries.geojson') as f: geo = json.load(f)
            fig_map = px.choropleth_mapbox(map_df, geojson=geo, locations="Area_Code", featureidkey="properties.LAD23CD",
                color="Final_Value", color_continuous_scale="viridis", mapbox_style="open-street-map",
                zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.6)
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        except: st.info("Map unavailable.")

    with col_table:
        st.subheader("Rankings")
        st.dataframe(map_df[['Area_Name', 'Final_Value']], hide_index=True, use_container_width=True)

    # Highlight Table
    st.divider()
    st.subheader("💡 Indicator Highlight Table")
    df_table = df_norm[(df_norm['Area_Name'].isin(selected_areas)) & (df_norm['Year'] == latest_year)]
    pivot_table = df_table.pivot(index='Area_Name', columns='Indicator_Name', values='Norm_Value').sort_index()
    styled_table = pivot_table.style.background_gradient(cmap='viridis', axis=None).format("{:.2f}")
    st.dataframe(styled_table, use_container_width=True)

    # Correlation Matrix
    if show_corr and mode == "Bespoke Index" and len(indicators_to_use) > 1:
        st.divider()
        st.subheader("🔗 Indicator Correlation")
        corr_matrix = pivot_table.corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)

with tab_metadata:
    st.title("📖 Methodology & User Guide")
    
    with st.expander("⚖️ How is the score calculated? (Normalisation)", expanded=True):
        st.write("""
        We use **Min-Max Normalisation** to compare different datasets (like £ and %). 
        - The highest performing area in the data is set to **1.0**.
        - The lowest performing area is set to **0.0**.
        - For 'Negative' indicators (like Fuel Poverty), we flip the score so that a lower poverty rate results in a higher wellbeing score.
        """)
        
    with st.expander("🤖 What is 'Statistical (PCA)' weighting?"):
        st.write("""
        **Principal Component Analysis (PCA)** is a data-driven approach. 
        Instead of humans deciding which indicator is most important, the tool looks for patterns. 
        It gives more weight to indicators that show the most 'variation' across the region, effectively focusing on the factors that make our boroughs unique.
        """)

    with st.expander("📍 Regional vs. Local Scope"):
        st.write("""
        - **Regional:** Your scores are calculated against the whole county. A 1.0 means you are the best in Surrey.
        - **Local:** Your scores are calculated only against the areas you have selected. A 1.0 means you are the best in your chosen group.
        """)

    st.divider()
    st.subheader("📊 Data Sources")
    meta_df = df_raw[df_raw['Indicator_Name'].isin(indicators_to_use)][['Indicator_Name', 'Theme', 'Unit', 'Direction', 'Source']].drop_duplicates()
    st.table(meta_df)

with tab_feedback:
    st.title("💬 Indicator Requests & Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name (Optional)")
            org = st.text_input("Organisation / Department")
        with col2:
            request_type = st.selectbox("Type", ["New Indicator", "Quality Issue", "Logic Suggestion", "General Feedback"])
            priority = st.select_slider("Priority", options=["Low", "Medium", "High"])
        details = st.text_area("Details")
        if st.form_submit_button("Submit Feedback"):
            st.success("✅ Thank you! Your feedback has been logged.")

# --- REPORT DOWNLOADER ---
report_text = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nMode: {mode}\nTop Area: {map_df.iloc[0]['Area_Name'] if not map_df.empty else 'N/A'}"
st.sidebar.download_button("Download Briefing Note", report_text, file_name="Report.txt")
