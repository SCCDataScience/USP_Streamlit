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
        df['Source'] = "ONS / Nomis"
    return df

try:
    df_raw = load_raw_data('USP_test.csv')
except FileNotFoundError:
    st.error("⚠️ CSV not found. Please upload to your GitHub repo.")
    st.stop()

# Load Map Boundaries
try:
    with open('boundaries.geojson') as f:
        geo = json.load(f)
except:
    geo = None

# --- GLOBAL SIDEBAR CONFIGURATION ---
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("What do you want to build?", [
    "1. Explore Single Indicator", 
    "2. View Existing Index", 
    "3. Build Bespoke Index", 
    "4. Compare Side-by-Side"
])

st.sidebar.divider()
st.sidebar.subheader("🌍 Geography & Scope")
all_areas = sorted(df_raw['Area_Name'].unique())
selected_areas = st.sidebar.multiselect("Select Areas to Display", all_areas, default=all_areas)
calc_scope = st.sidebar.radio("Normalisation Scope", ["Regional (All Surrey)", "Local (Selected only)"])

# Filter Scope for Math
df_calc = df_raw[df_raw['Area_Name'].isin(selected_areas)].copy() if calc_scope == "Local (Selected only)" else df_raw.copy()

# --- MAIN INTERFACE (TABS) ---
tab_dashboard, tab_metadata, tab_feedback = st.tabs(["📊 Dashboard", "📖 Data Sources & Help", "💬 Feedback"])

with tab_dashboard:
    st.title("🏙️ Strategic Places & Wellbeing Explorer")
    
    # ==========================================
    # MODE 1: SINGLE INDICATOR (Updated with Sub-Categories)
    # ==========================================
    if app_mode == "1. Explore A Single Indicator":
        st.subheader("Explore Single Indicator")
        selected_ind = st.sidebar.selectbox("Select Indicator", sorted(df_raw['Indicator_Name'].unique()))
        
        display_data = df_calc[df_calc['Indicator_Name'] == selected_ind]
        display_data = display_data[display_data['Area_Name'].isin(selected_areas)]
        latest_year = display_data['Year'].max()
        map_df = display_data[display_data['Year'] == latest_year]
        
        # --- Check for Sub-Categories ---
        has_sub_categories = False
        if 'Sub_Category' in display_data.columns:
            # Drop empty/null sub-categories to see if actual breakdown exists
            valid_subs = display_data['Sub_Category'].dropna().unique()
            if len(valid_subs) > 0:
                has_sub_categories = True
        
        if has_sub_categories:
            st.markdown(f"**Composition Breakdown ({latest_year})**")
            # Stacked Bar Chart
            fig_bar = px.bar(map_df, x="Area_Name", y="Value", color="Sub_Category", barmode="stack",
                             labels={"Value": map_df['Unit'].iloc[0] if not map_df.empty else "Value"},
                             color_discrete_sequence=px.colors.qualitative.Prism)
            fig_bar.update_layout(xaxis_title="Area", yaxis_title="Percentage / Value")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("💡 **Note:** Because this indicator is a composition of multiple sub-categories, it is visualised as a stacked bar rather than a time-series trend line.")
        else:
            # Standard Line Chart
            st.markdown("**Performance Trend**")
            fig_line = px.line(display_data, x="Year", y="Value", color="Area_Name", markers=True)
            fig_line.update_layout(xaxis_type='category')
            st.plotly_chart(fig_line, use_container_width=True)
        
        # --- Map and Table Rendering ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Map View ({latest_year})**")
            if has_sub_categories:
                st.warning("Maps require a single aggregate value. Sub-category mapping will be supported in a future update.")
            elif geo:
                fig_map = px.choropleth_mapbox(map_df, geojson=geo, locations="Area_Name", featureidkey="properties.LAD23NM",
                    color="Value", color_continuous_scale="viridis", mapbox_style="open-street-map",
                    zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.6)
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
        with col2:
            st.markdown("**Current Data Table**")
            st.dataframe(map_df[['Area_Name', 'Value'] + (['Sub_Category'] if has_sub_categories else [])].sort_values('Value', ascending=False), hide_index=True)

    # ==========================================
    # MODE 2: EXISTING INDEX
    # ==========================================
    elif app_mode == "2. View An Existing Index":
        st.subheader("Existing Strategic Indices")
        existing_index = st.selectbox("Select Existing Index", ["Surrey Index (Mock)", "Health and Wellbeing Strategy Index (Mock)"])
        st.write(f"Currently viewing the architecture for the **{existing_index}**.")

    # ==========================================
    # MODE 3: BESPOKE INDEX
    # ==========================================
    elif app_mode == "3. Build A Bespoke Index":
        st.subheader("Bespoke Index Builder")
        with st.sidebar.expander("📊 Index Components", expanded=True):
            selected_inds = st.multiselect("Select Indicators", sorted(df_raw['Indicator_Name'].unique()), default=sorted(df_raw['Indicator_Name'].unique())[:4])
        weight_method = st.sidebar.selectbox("Weighting Logic", ["Equal Weighting", "Statistical (PCA)"])

        normalized_list = []
        for ind in selected_inds:
            subset = df_calc[df_calc['Indicator_Name'] == ind].copy()
            if not subset.empty:
                subset['Norm_Value'] = normalize_series(subset['Value'], subset['Direction'].iloc[0])
                normalized_list.append(subset)
        
        if normalized_list:
            df_norm = pd.concat(normalized_list)
            t_score = df_norm.groupby(['Area_Name', 'Year'])['Norm_Value'].mean().reset_index().rename(columns={'Norm_Value': 'Final_Value'})
            
            display_data = t_score[t_score['Area_Name'].isin(selected_areas)]
            latest_year = display_data['Year'].max()
            map_df = display_data[display_data['Year'] == latest_year]

            col1, col2 = st.columns([2, 1])
            with col1:
                if geo:
                    fig_map = px.choropleth_mapbox(map_df, geojson=geo, locations="Area_Name", featureidkey="properties.LAD23NM",
                        color="Final_Value", color_continuous_scale="viridis", mapbox_style="open-street-map",
                        zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.6)
                    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)
            with col2:
                st.dataframe(map_df[['Area_Name', 'Final_Value']].sort_values('Final_Value', ascending=False), hide_index=True)
                
            st.divider()
            show_corr = st.toggle("View Correlation Matrix?")
            if show_corr and len(selected_inds) > 1:
                pivot_table = df_norm[df_norm['Year'] == latest_year].pivot(index='Area_Name', columns='Indicator_Name', values='Norm_Value')
                st.dataframe(pivot_table.corr().style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.2f}"))

    # ==========================================
    # MODE 4: SIDE-BY-SIDE
    # ==========================================
    elif app_mode == "4. Compare Side-by-Side":
        st.subheader("Indicator Correlation & Spatial Comparison")
        
        ind_a = st.sidebar.selectbox("Indicator A (Left Map)", sorted(df_raw['Indicator_Name'].unique()), index=0)
        ind_b = st.sidebar.selectbox("Indicator B (Right Map)", sorted(df_raw['Indicator_Name'].unique()), index=1 if len(df_raw['Indicator_Name'].unique()) > 1 else 0)

        df_a = df_calc[(df_calc['Indicator_Name'] == ind_a) & (df_calc['Area_Name'].isin(selected_areas))]
        df_b = df_calc[(df_calc['Indicator_Name'] == ind_b) & (df_calc['Area_Name'].isin(selected_areas))]
        
        latest_year_a = df_a['Year'].max()
        latest_year_b = df_b['Year'].max()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{ind_a}** ({latest_year_a})")
            if geo:
                fig_a = px.choropleth_mapbox(df_a[df_a['Year'] == latest_year_a], geojson=geo, locations="Area_Name", featureidkey="properties.LAD23NM",
                    color="Value", color_continuous_scale="Blues", mapbox_style="open-street-map",
                    zoom=8.5, center={"lat": 51.3, "lon": -0.4}, opacity=0.7)
                fig_a.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_a, use_container_width=True)

        with col2:
            st.markdown(f"**{ind_b}** ({latest_year_b})")
            if geo:
                fig_b = px.choropleth_mapbox(df_b[df_b['Year'] == latest_year_b], geojson=geo, locations="Area_Name", featureidkey="properties.LAD23NM",
                    color="Value", color_continuous_scale="Reds", mapbox_style="open-street-map",
                    zoom=8.5, center={"lat": 51.3, "lon": -0.4}, opacity=0.7)
                fig_b.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_b, use_container_width=True)
            
with tab_metadata:
    st.write("Methodology and definitions live here.")
    
with tab_feedback:
    st.write("User feedback form lives here.")
