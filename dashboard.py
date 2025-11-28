# ==========================================
# Interactive Dashboard (Final Polish)
# Features: Upright 3D Globe, Click-Event, Full Screen
# ==========================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# 1. Page Config
st.set_page_config(
    page_title="Social Trends AI", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Data Loading
@st.cache_data
def load_data():
    paths = ["./data/final_report_data.csv", "final_report_data.csv", "data/final_report_data.csv"]
    file_path = None
    for p in paths:
        if os.path.exists(p):
            file_path = p
            break
    
    if not file_path: return pd.DataFrame()

    df = pd.read_csv(file_path)
    if 'Country' not in df.columns:
        if df.iloc[:, 0].dtype == object: 
            df.rename(columns={df.columns[0]: 'Country'}, inplace=True)
    return df

df = load_data()

if df.empty:
    st.error("❌ Error: Data not found. Run Step 3 ML first.")
    st.stop()

# 3. Backend Model
try:
    if 'Log_GDP' not in df.columns: df['Log_GDP'] = np.log(df['GDP_per_Capita'])
    
    life_col = 'Health'
    for col in ['lifespan', 'Health', 'Healthy life expectancy']:
        if col in df.columns: life_col = col; break
            
    X = df[['Log_GDP', 'Education_Expenditure', life_col]]
    y = df['Score']
    model = LinearRegression()
    model.fit(X, y)
    df['Predicted_Score'] = model.predict(X)
except:
    df['Predicted_Score'] = df['Score']

# ==========================================
# 4. Interaction Logic
# ==========================================
if 'target_country' not in st.session_state:
    st.session_state.target_country = sorted(df['Country'].unique())[0]

def update_from_sidebar():
    st.session_state.target_country = st.session_state.sidebar_selection

# ==========================================
# 5. The 3D Globe (Upright Fixed)
# ==========================================
st.title("🌍 Global Social Trends & AI Analytics")

# Create 3D Map
fig_map = px.choropleth(
    df, 
    locations="Country", 
    locationmode="country names",
    color="Score",
    hover_name="Country",
    color_continuous_scale='Plasma',
    title="", 
    labels={'Score': 'Happiness'},
    projection="orthographic"
)

# --- VISUAL TWEAKS ---
fig_map.update_layout(
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0},
    geo=dict(
        showframe=False,
        showcoastlines=False,
        bgcolor='rgba(0,0,0,0)', 
        showocean=True,
        oceancolor="#1E1E1E",
        projection_scale=1.0,
        # 🛡️ FORCE UPRIGHT ORIENTATION
        # This ensures the globe starts perfectly vertical
        projection_rotation=dict(lon=0, lat=0, roll=0) 
    ),
)

# *** INTERACTIVE EVENT ***
selection = st.plotly_chart(
    fig_map, 
    use_container_width=True, 
    on_select="rerun", 
    selection_mode="points",
    # Disable scroll zoom to prevent accidental resizing
    config={'scrollZoom': False, 'displayModeBar': False} 
)

# Handle Click
if selection and len(selection['selection']['points']) > 0:
    clicked_idx = selection['selection']['points'][0]['point_index']
    st.session_state.target_country = df.iloc[clicked_idx]['Country']

# Sync Data
current_country = st.session_state.target_country
country_data = df[df['Country'] == current_country].iloc[0]

# ==========================================
# 6. Analysis Dashboard
# ==========================================
st.markdown(f"### 👇 Analyzing: **{current_country}**")
st.divider()

# Sidebar Sync
country_list = sorted(df['Country'].astype(str).unique())
try: list_index = country_list.index(current_country)
except: list_index = 0

with st.sidebar:
    st.header("Navigate")
    st.selectbox("Select Country:", country_list, index=list_index, key='sidebar_selection', on_change=update_from_sidebar)
    st.info("💡 Tip: Click the globe to select! (Free rotation is a Plotly feature)")

# Layout: Metrics + Radar + Insights
c1, c2, c3, c4 = st.columns(4)
c1.metric("😊 Happiness", f"{country_data['Score']:.2f}")
c2.metric("💰 GDP", f"${country_data['GDP_per_Capita']:,.0f}")
c3.metric("🎓 Education", f"{country_data['Education_Expenditure']:.1f}%")
c4.metric("❤️ Lifespan", f"{country_data[life_col]:.1f} yrs")

col_L, col_R = st.columns([1, 1])

with col_L:
    # Radar Chart
    def norm(val, col):
        mn, mx = df[col].min(), df[col].max()
        return (val - mn) / (mx - mn) if mx > mn else 0

    vals = [
        norm(country_data['Log_GDP'], 'Log_GDP'),
        norm(country_data['Education_Expenditure'], 'Education_Expenditure'),
        norm(country_data[life_col], life_col)
    ]
    avgs = [
        norm(df['Log_GDP'].mean(), 'Log_GDP'),
        norm(df['Education_Expenditure'].mean(), 'Education_Expenditure'),
        norm(df[life_col].mean(), life_col)
    ]
    
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=vals, theta=['Economy', 'Education', 'Health'], fill='toself', name=current_country))
    radar.add_trace(go.Scatterpolar(r=avgs, theta=['Economy', 'Education', 'Health'], name='Global Avg', line_dash='dash', line_color='grey'))
    
    # Theme-adaptive colors
    radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=350, 
        margin=dict(t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="gray")
    )
    st.plotly_chart(radar, use_container_width=True)

with col_R:
    # AI Insights
    cluster = country_data.get('Cluster_Label', 'Unknown')
    pred = country_data.get('Predicted_Score', 0)
    
    if cluster == 'Developed':
        color, advice = "green", "Focus on **mental well-being** and **work-life balance**."
    elif cluster == 'Developing':
        color, advice = "orange", "Prioritize **education quality** and **healthcare access**."
    else:
        color, advice = "red", "Urgent need for **basic infrastructure** and **poverty alleviation**."

    st.markdown(f"""
    #### 🤖 AI Insight
    * **Cluster**: :{color}[{cluster}]
    * **Prediction**: Model expects **{pred:.2f}**, Actual is **{country_data['Score']:.2f}**.
    * **Strategy**: {advice}
    """)
    st.caption(f"Analysis based on K-Means clustering (Tier) and Linear Regression (Prediction).")