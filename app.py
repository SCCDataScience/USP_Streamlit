import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime

# 1. Page Configuration
#st.set_page_config(page_title="Understanding Surrey's Places", layout="wide")

# --- CUSTOM HEADER BANNER ---
st.markdown("""
    <style>
    .title-text {
        color: #1A5632; /* Dark Green */
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    try:
        # Ensure the image is uploaded to GitHub as surrey_logo.png
        st.image("surrey_logo.png", use_container_width=True)
    except FileNotFoundError:
        st.warning("Logo missing. Please upload surrey_logo.png")

with col2:
    st.markdown('<p class="title-text">Surrey Data Explorer</p>', unsafe_allow_html=True)

st.divider()

# --- CORE MATH FUNCTIONS ---
def normalize_series(series, direction):
    v_min, v_max = series.min(), series.max()
    if v_max == v_min: return series * 0 + 0.5
    norm = (series - v_min) / (v_max - v_min)
    return 1 - norm if direction == 'Negative' else norm

def apply_pca_weights(df_pivot):
    try:
        pca = PCA(n_components=1)
        pca.fit(df_pivot)
        weights = np.abs(pca.components_[0])
        return weights / weights.sum()
    except: return np.array([1/df_pivot.shape[1]] * df_pivot.shape[1])

# --- DATA LOADING ---
import os

@st.cache_data
def load_raw_data(file_path):
    df = pd.read_csv(file_path)
    
    # Delete completely blank rows left by Excel
    df = df.dropna(how='all')
    
    # 1. Strip invisible spaces from names so they match the map perfectly
    df['Indicator_Name'] = df['Indicator_Name'].astype(str).str.strip()
    df['Area_Name'] = df['Area_Name'].astype(str).str.strip()
    
    # 2. Force the 'Value' column to be numeric (removes commas, handles errors)
    if df['Value'].dtype == object:
        df['Value'] = df['Value'].astype(str).str.replace(',', '')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    if 'Source' not in df.columns: 
        df['Source'] = "ONS / Nomis"
    return df

@st.cache_data
def load_mock_services():
    return pd.DataFrame({
        "Service_Name": ["Ashford Hospital", "Royal Surrey", "Woking High", "Guildford Library", "Elmbridge Leisure", "Epsom General"],
        "Type": ["Health", "Health", "Education", "Public Service", "Public Service", "Health"],
        "Lat": [51.426, 51.240, 51.325, 51.236, 51.370, 51.326],
        "Lon": [-0.473, -0.602, -0.560, -0.575, -0.410, -0.270]
    })

try: 
    df_raw = load_raw_data('USP_test.csv')
except FileNotFoundError: 
    st.error("⚠️ 'USP_test.csv' not found."); st.stop()

df_services = load_mock_services()

# FIX 2: Loud Map Diagnostics
geo = None
if os.path.exists('boundaries.geojson'):
    with open('boundaries.geojson') as f: 
        geo = json.load(f)
elif os.path.exists('boundaries.json'):
    st.warning("⚠️ Map Warning: Found 'boundaries.json'. Please rename the file to end in '.geojson' on GitHub.")
    with open('boundaries.json') as f: 
        geo = json.load(f)
else:
    st.error("⚠️ Map Error: Could not find the boundaries file. Please ensure it is uploaded to the main folder of your GitHub repository.")

# --- GLOBAL SIDEBAR CONFIGURATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("What do you want to build?", [
    "1. Explore A Single Indicator", 
    "2. View An Existing Index", 
    "3. Build A Bespoke Index", 
    "4. Compare Side-by-Side",
    "5. Spatial Correlation (Bivariate Map)",
    "6. Local Services Mapper (Pins)"
])

st.sidebar.divider()
st.sidebar.subheader("Geography & Scope")
all_areas = sorted(df_raw['Area_Name'].unique())
selected_areas = st.sidebar.multiselect("Select Areas to Display", all_areas, default=all_areas)
calc_scope = st.sidebar.radio("Normalisation Scope", ["Regional (All Surrey)", "Local (Selected only)"])

df_calc = df_raw[df_raw['Area_Name'].isin(selected_areas)].copy() if calc_scope == "Local (Selected only)" else df_raw.copy()

# --- MAIN INTERFACE (TABS) ---
tab_dashboard, tab_metadata, tab_feedback = st.tabs(["Dashboard", "Data Sources & Help", "Feedback"])

#with tab_dashboard:
    #st.title("Understanding Surrey's Places")
    
   # ==========================================
    # MODE 1: SINGLE INDICATOR (Sub-Category Fix)
    # ==========================================
    if app_mode == "1. Explore A Single Indicator":
        st.subheader("Explore Single Indicator")
        selected_ind = st.sidebar.selectbox("Select Indicator", sorted(df_raw['Indicator_Name'].unique()))
        
        display_data = df_calc[df_calc['Indicator_Name'] == selected_ind]
        display_data = display_data[display_data['Area_Name'].isin(selected_areas)]
        latest_year = display_data['Year'].max()
        
        has_sub_categories = 'Sub_Category' in display_data.columns and len(display_data['Sub_Category'].dropna().unique()) > 0
        
        if has_sub_categories:
            st.markdown(f"**Composition Breakdown ({latest_year})**")
            map_df = display_data[display_data['Year'] == latest_year]
            fig_bar = px.bar(map_df, x="Area_Name", y="Value", color="Sub_Category", barmode="stack", color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            st.markdown("### **Trend & Map View**")
            selected_sub = st.selectbox("Select Sub-Category to Map and Track:", display_data['Sub_Category'].dropna().unique())
            active_data = display_data[display_data['Sub_Category'] == selected_sub]
        else:
            active_data = display_data
            
        map_df_active = active_data[active_data['Year'] == latest_year]

        # Trend & Map Render
        fig_line = px.line(active_data, x="Year", y="Value", color="Area_Name", markers=True)
        fig_line.update_layout(xaxis_type='category')
        st.plotly_chart(fig_line, use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if geo:
                fig_map = px.choropleth_mapbox(map_df_active, geojson=geo, locations="Area_Name", featureidkey="properties.LAD23NM",
                    color="Value", color_continuous_scale="viridis", mapbox_style="open-street-map",
                    zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.6)
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
        with col2:
            st.dataframe(map_df_active[['Area_Name', 'Value']].sort_values('Value', ascending=False), hide_index=True)

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

    # ==========================================
    # MODE 5: SPATIAL CORRELATION (BIVARIATE MAP)
    # ==========================================
    elif app_mode == "5. Spatial Correlation (Bivariate Map)":
        st.subheader("Bivariate Correlation Map")
        st.write("This map blends two indicators to identify complex spatial relationships (e.g., areas with High Health Deprivation AND Low Skills).")
        
        col1, col2 = st.columns(2)
        with col1: ind_x = st.selectbox("Indicator 1 (X-Axis)", sorted(df_raw['Indicator_Name'].unique()), index=0)
        with col2: ind_y = st.selectbox("Indicator 2 (Y-Axis)", sorted(df_raw['Indicator_Name'].unique()), index=1)

        # Get latest data
        df_x = df_calc[df_calc['Indicator_Name'] == ind_x].sort_values('Year').groupby('Area_Name').last().reset_index()
        df_y = df_calc[df_calc['Indicator_Name'] == ind_y].sort_values('Year').groupby('Area_Name').last().reset_index()
        
        biv_df = pd.merge(df_x[['Area_Name', 'Value']], df_y[['Area_Name', 'Value']], on='Area_Name', suffixes=('_X', '_Y'))
        
        # Calculate Terciles (3x3 Grid) safely handling low data points
        try:
            biv_df['X_Quant'] = pd.qcut(biv_df['Value_X'], 3, labels=['1', '2', '3'], duplicates='drop').astype(str)
            biv_df['Y_Quant'] = pd.qcut(biv_df['Value_Y'], 3, labels=['1', '2', '3'], duplicates='drop').astype(str)
            biv_df['Biv_Class'] = biv_df['X_Quant'] + "-" + biv_df['Y_Quant']
            
            # Standard Tequila/Pink Bivariate Palette
            biv_colors = {"3-3": "#3F2949", "2-3": "#435786", "1-3": "#4885C1", "3-2": "#77324C", "2-2": "#806A8A", "1-2": "#89A1C8", "3-1": "#AE3A4E", "2-1": "#BC7C8F", "1-1": "#CABED0"}
            
            if geo:
                fig_biv = px.choropleth_mapbox(biv_df, geojson=geo, locations="Area_Name", featureidkey="properties.LAD23NM",
                    color="Biv_Class", color_discrete_map=biv_colors, mapbox_style="open-street-map",
                    zoom=9, center={"lat": 51.3, "lon": -0.4}, opacity=0.8, hover_data={"Biv_Class": False, "Area_Name": True, "Value_X": True, "Value_Y": True})
                fig_biv.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
                st.plotly_chart(fig_biv, use_container_width=True)
                
            st.info("**How to read this map:** Dark Purple (`3-3`) = High in both. Light Grey (`1-1`) = Low in both. Bright Red (`3-1`) = High Ind 1, Low Ind 2. Bright Blue (`1-3`) = Low Ind 1, High Ind 2.")
        except Exception as e:
            st.warning("Not enough variance in the selected data to create a 3x3 statistical grid. Try different indicators.")

    # ==========================================
    # MODE 6: LOCAL SERVICES MAPPER (PINS)
    # ==========================================
    elif app_mode == "6. Local Services Mapper (Pins)":
        st.subheader("Asset & Services Mapper")
        st.write("Overlay local infrastructure (Schools, GP Surgeries, Libraries) on top of deprivation or economic data.")
        
        base_ind = st.selectbox("Select Background Heatmap Layer", sorted(df_raw['Indicator_Name'].unique()))
        selected_types = st.multiselect("Select Services to Display", df_services['Type'].unique(), default=df_services['Type'].unique())
        
        df_base = df_calc[df_calc['Indicator_Name'] == base_ind]
        map_df_base = df_base[df_base['Year'] == df_base['Year'].max()]
        filtered_services = df_services[df_services['Type'].isin(selected_types)]

        if geo:
            # We use Graph Objects (go) to combine two different map layers
            fig = go.Figure()
            
            # Layer 1: The Borough Polygons
            fig.add_trace(go.Choroplethmapbox(
                geojson=geo, locations=map_df_base['Area_Name'], featureidkey="properties.LAD23NM",
                z=map_df_base['Value'], colorscale="Blues", marker_opacity=0.5,
                name="Heatmap", hoverinfo="location+z"
            ))
            
            # Layer 2: The Service Pins
            color_map = {"Health": "red", "Education": "green", "Public Service": "orange"}
            for s_type in selected_types:
                type_data = filtered_services[filtered_services['Type'] == s_type]
                fig.add_trace(go.Scattermapbox(
                    lat=type_data['Lat'], lon=type_data['Lon'], mode='markers',
                    marker=go.scattermapbox.Marker(size=12, color=color_map.get(s_type, "blue")),
                    text=type_data['Service_Name'], hoverinfo='text', name=s_type
                ))

            fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=9, mapbox_center={"lat": 51.3, "lon": -0.4}, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

with tab_metadata:
    st.write("Methodology and definitions live here.")
    
with tab_feedback:
    st.write("User feedback form lives here.")
