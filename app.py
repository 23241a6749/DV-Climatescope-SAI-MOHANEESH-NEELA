import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ——————————————————————————————————————————————
# Utility Functions
# ——————————————————————————————————————————————

def simple_linregress(x, y):
    x, y = np.array(x), np.array(y)
    slope = np.sum((x - x.mean())*(y - y.mean())) / np.sum((x - x.mean())**2)
    intercept = y.mean() - slope * x.mean()
    return slope, intercept

def get_climate_insights(data, variable, country):
    """Generate warming/cooling insights for one country & variable."""
    insights = []
    trend = data.groupby('year')[variable].mean()
    if len(trend) > 1:
        slope, _ = simple_linregress(trend.index, trend.values)
        if variable=='temperature_celsius':
            if slope>0.1: insights.append(f"🌡️ {country} warming +{slope:.2f}°C/yr")
            elif slope<-0.05: insights.append(f"❄️ {country} cooling {slope:.2f}°C/yr")
        elif variable=='precip_mm':
            if slope>5: insights.append(f"🌧️ {country} wetter +{slope:.1f}mm/yr")
            elif slope<-5: insights.append(f"🏜️ {country} drier {slope:.1f}mm/yr")
        elif variable=='humidity':
            if slope>1: insights.append(f"💧 {country} more humid +{slope:.1f}%/yr")
            elif slope<-1: insights.append(f"🏜️ {country} less humid {slope:.1f}%/yr")
        elif variable=='wind_kph':
            if slope>0.5: insights.append(f"💨 {country} windier +{slope:.1f} kph/yr")
            elif slope<-0.5: insights.append(f"🌊 {country} calmer winds {slope:.1f} kph/yr")
    return insights

# ——————————————————————————————————————————————
# Data Loading
# ——————————————————————————————————————————————

@st.cache_data
def load_data():
    df = pd.read_csv("monthly_weather_aggregated.csv")
    df["year"]  = df["year_month"].str.split("-").str[0].astype(int)
    df["month"] = df["year_month"].str.split("-").str[1].astype(int)
    df["date"]  = pd.to_datetime(df["year_month"] + "-01")
    return df

df = load_data()

# ——————————————————————————————————————————————
# Page & Styles
# ——————————————————————————————————————————————

st.set_page_config(page_title="ClimateScope", page_icon="🌍", layout="wide")
st.markdown("""
<style>
body { background: #f4f7fa; }
.stButton>button { background: #2a5298; color:white; }
</style>
""", unsafe_allow_html=True)

# ——————————————————————————————————————————————
# Header
# ——————————————————————————————————————————————

st.markdown("""
<div style="text-align:center; padding:20px; background:linear-gradient(90deg,#1e3c72,#2a5298); border-radius:10px;">
  <h1 style="color:white;">🌍 ClimateScope Dashboard</h1>
  <p style="color:#ddd;">Interactive climate visualization & insights</p>
</div>
""", unsafe_allow_html=True)

# ——————————————————————————————————————————————
# Sidebar Controls
# ——————————————————————————————————————————————

vars_map = {
    "🌡️ Temperature (°C)":"temperature_celsius",
    "🌧️ Precipitation (mm)":"precip_mm",
    "💧 Humidity (%)":"humidity",
    "💨 Wind Speed (kph)":"wind_kph"
}

countries_all = sorted(df.country.unique())
st.sidebar.header("Filters")
sel_countries = st.sidebar.multiselect("Countries", countries_all, default=countries_all[:4])
sel_var       = st.sidebar.selectbox("Variable", list(vars_map.keys()))
yr_min, yr_max= int(df.year.min()), int(df.year.max())
sel_years     = st.sidebar.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
baseline      = st.sidebar.slider("Baseline Period", yr_min, yr_max, (yr_min, min(yr_min+9, yr_max)))
show_trend    = st.sidebar.checkbox("Show Trend Lines", True)
show_anom     = st.sidebar.checkbox("Show Anomalies", True)
show_extreme  = st.sidebar.checkbox("Highlight Extremes", True)
pct_extreme   = st.sidebar.slider("Extreme %ile", 90, 99, 95)
if st.sidebar.button("Export CSV"):
    tmp = df[df.country.isin(sel_countries)&df.year.between(*sel_years)]
    st.sidebar.download_button("Download", tmp.to_csv(index=False), "data.csv")

if not sel_countries:
    st.warning("Select at least one country."); st.stop()

# ——————————————————————————————————————————————
# Filter Data
# ——————————————————————————————————————————————

var_col = vars_map[sel_var]
filtered = df[df.country.isin(sel_countries)&df.year.between(*sel_years)]
if filtered.empty:
    st.warning("No data for filters."); st.stop()

baseline_means = df[df.year.between(*baseline)].groupby("country")[var_col].mean()
filtered["anomaly"] = filtered.apply(lambda r: r[var_col]-baseline_means.get(r.country,0), axis=1)

# ——————————————————————————————————————————————
# Tabs
# ——————————————————————————————————————————————

tab1, tab2, tab3 = st.tabs(["🗺️ Map","📈 Trends","🔍 Analysis"])

with tab1:
    st.subheader("Average Values Map")
    agg = filtered.groupby("country")[var_col].mean().reset_index()
    fig = px.choropleth(agg, locations="country", locationmode="country names",
                        color=var_col, color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Time Series")
    fig = px.line(filtered, x="date", y=var_col, color="country", markers=True)
    if show_trend:
        for c in sel_countries:
            d = filtered[filtered.country==c].groupby("date")[var_col].mean().reset_index()
            s,i = simple_linregress(d.date.map(datetime.toordinal), d[var_col])
            fig.add_trace(go.Scatter(x=d.date, y=i+s*d.date.map(datetime.toordinal),
                                     mode="lines", line=dict(dash="dash"), name=f"{c} Trend"))
    if show_anom:
        an = filtered[np.abs(filtered.anomaly)>filtered.anomaly.std()]
        fig.add_trace(go.Scatter(x=an.date, y=an[var_col],
                                 mode="markers", marker=dict(color="red",size=6), name="Anomaly"))
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Distribution & Insights")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Distribution**")
        st.plotly_chart(px.box(filtered, x="country", y=var_col), use_container_width=True)
        st.markdown("**Correlation**")
        corr = df[list(vars_map.values())].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)
    with c2:
        st.markdown("**Extreme Events**")
        hi = filtered[filtered[var_col]>=filtered[var_col].quantile(pct_extreme/100)]
        lo = filtered[filtered[var_col]<=filtered[var_col].quantile((100-pct_extreme)/100)]
        st.write(f"High: {len(hi)}  Low: {len(lo)}")
        ext_fig = go.Figure()
        ext_fig.add_trace(go.Scatter(x=hi.date, y=hi[var_col], mode="markers", marker=dict(color="red"), name="High"))
        ext_fig.add_trace(go.Scatter(x=lo.date, y=lo[var_col], mode="markers", marker=dict(color="blue"), name="Low"))
        st.plotly_chart(ext_fig, use_container_width=True)
    st.markdown("**Smart Insights**")
    for c in sel_countries:
        for insight in get_climate_insights(filtered[filtered.country==c], var_col, c):
            st.info(insight)
