
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Simple linear regression function using numpy (replaces scipy.stats.linregress)
def simple_linregress(x, y):
    """Simple linear regression using numpy"""
    x = np.array(x)
    y = np.array(y)

    # Calculate slope and intercept
    slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
    intercept = np.mean(y) - slope * np.mean(x)

    return slope, intercept

def get_climate_insights(data, variable, country):
    """Generate educational insights about climate data"""
    insights = []

    avg_val = data[variable].mean()
    trend_data = data.groupby('year')[variable].mean()

    if len(trend_data) > 1:
        slope, _ = simple_linregress(trend_data.index, trend_data.values)

        if variable == 'temperature_celsius':
            if slope > 0.1:
                insights.append(f"🌡️ {country} is experiencing significant warming trend (+{slope:.2f}°C per year)")
            elif slope > 0.05:
                insights.append(f"🌡️ {country} shows moderate warming trend (+{slope:.2f}°C per year)")
            elif slope < -0.05:
                insights.append(f"❄️ {country} shows cooling trend ({slope:.2f}°C per year)")

        elif variable == 'precip_mm':
            if slope > 5:
                insights.append(f"🌧️ {country} is getting wetter (+{slope:.1f}mm per year)")
            elif slope < -5:
                insights.append(f"🏜️ {country} is getting drier ({slope:.1f}mm per year)")

        elif variable == 'humidity':
            if slope > 1:
                insights.append(f"💧 {country} is becoming more humid (+{slope:.1f}% per year)")
            elif slope < -1:
                insights.append(f"🏜️ {country} is becoming drier ({slope:.1f}% per year)")

        elif variable == 'wind_kph':
            if slope > 0.5:
                insights.append(f"💨 {country} is getting windier (+{slope:.1f} kph per year)")
            elif slope < -0.5:
                insights.append(f"🌊 {country} winds are calming ({slope:.1f} kph per year)")

    return insights

def get_climate_education(variable):
    """Provide educational content about climate variables"""
    education = {
        'temperature_celsius': {
            'definition': 'Temperature measures how hot or cold the air is. It affects weather patterns, ecosystems, and human activities.',
            'importance': 'Rising global temperatures indicate climate change, affecting sea levels, weather patterns, and biodiversity.',
            'normal_range': 'Global average: 14-15°C. Varies by location and season.',
            'impacts': ['Affects plant growth cycles', 'Influences weather patterns', 'Impacts human health', 'Changes wildlife habitats'],
            'facts': [
                'Global temperatures have risen ~1.1°C since pre-industrial times',
                'The last decade was the warmest on record',
                'Arctic warming is happening twice as fast as the global average',
                'Temperature affects 70% of all biological processes'
            ]
        },
        'precip_mm': {
            'definition': 'Precipitation is water falling from the sky as rain, snow, sleet, or hail, measured in millimeters.',
            'importance': 'Precipitation patterns affect water supply, agriculture, and natural disasters like floods and droughts.',
            'normal_range': 'Global average: ~990mm/year. Varies greatly by region (0-11,000mm annually).',
            'impacts': ['Determines water availability', 'Affects crop yields', 'Influences flood/drought risk', 'Shapes ecosystems'],
            'facts': [
                'Only 3% of Earth\'s water is fresh water',
                'Climate change is making wet areas wetter and dry areas drier',
                '1 billion people lack access to clean water',
                'Agriculture uses 70% of global freshwater'
            ]
        },
        'humidity': {
            'definition': 'Humidity measures the amount of water vapor in the air, expressed as a percentage.',
            'importance': 'Humidity affects comfort, health, weather formation, and energy consumption.',
            'normal_range': 'Comfortable range: 30-60%. Tropical regions often >70%, deserts <30%.',
            'impacts': ['Affects human comfort', 'Influences weather formation', 'Impacts building energy use', 'Affects air quality'],
            'facts': [
                'Humidity above 60% can feel uncomfortable for humans',
                'Low humidity increases risk of respiratory infections',
                'Humidity affects how hot temperatures feel to humans',
                'High humidity reduces the body\'s ability to cool through sweating'
            ]
        },
        'wind_kph': {
            'definition': 'Wind speed measures how fast air is moving, typically in kilometers per hour.',
            'importance': 'Wind drives weather systems, affects temperature distribution, and is a renewable energy source.',
            'normal_range': 'Global average: 3-5 m/s (11-18 kph). Storm winds: >60 kph.',
            'impacts': ['Transports weather systems', 'Affects temperature distribution', 'Renewable energy source', 'Influences air quality'],
            'facts': [
                'Wind power could provide 20% of global electricity by 2030',
                'Jet streams are rivers of fast-moving air 10km above Earth',
                'Trade winds helped early ocean exploration',
                'Wind erosion shapes landscapes over thousands of years'
            ]
        }
    }
    return education.get(variable, {})

def get_climate_tips(variable):
    """Get actionable climate tips based on the variable"""
    tips = {
        'temperature_celsius': [
            '🏠 Improve home insulation to reduce heating/cooling needs',
            '🌳 Plant trees for natural cooling and carbon absorption',
            '☀️ Use solar panels to reduce fossil fuel dependence',
            '🚗 Choose electric or hybrid vehicles'
        ],
        'precip_mm': [
            '💧 Install rainwater harvesting systems',
            '🌱 Use drought-resistant plants in your garden',
            '🚿 Fix leaks and use water-efficient appliances',
            '🌾 Support sustainable agriculture practices'
        ],
        'humidity': [
            '🏠 Use energy-efficient dehumidifiers when needed',
            '🌿 Choose plants that naturally regulate humidity',
            '🪟 Improve ventilation to maintain comfortable humidity',
            '❄️ Use smart thermostats to optimize energy use'
        ],
        'wind_kph': [
            '💨 Support wind energy projects in your community',
            '🏠 Use wind-resistant building designs in windy areas',
            '🌪️ Prepare emergency plans for extreme wind events',
            '🌊 Protect coastal areas with natural windbreaks'
        ]
    }
    return tips.get(variable, [])

# Load processed data
try:
    df = pd.read_csv('monthly_weather_aggregated.csv')
    df['year'] = df['year_month'].str.split('-').str[0].astype(int)
    df['month'] = df['year_month'].str.split('-').str[1].astype(int)
except FileNotFoundError:
    st.error("📁 Data file not found. Please ensure 'monthly_weather_aggregated.csv' is in the correct location.")
    st.info("Make sure you have completed Milestone 1 and created the aggregated data file.")
    st.stop()

st.set_page_config(layout="wide", page_title="ClimateScope Educational Dashboard", page_icon="🌍")

# Header with branding
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; margin-bottom: 30px; border-radius: 10px;'>
    <h1>🌍 ClimateScope: Educational Climate Dashboard</h1>
    <h3><i>Explore, Learn, and Understand Global Weather Patterns</i></h3>
    <p>Your gateway to climate knowledge and environmental awareness</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced controls
st.sidebar.markdown("## 🎛️ Dashboard Controls")

# Get available countries and set smart defaults
available_countries = sorted(df['country'].unique())
default_countries = []
preferred_defaults = ['USA', 'India', 'Brazil', 'China', 'Germany', 'Australia', 'Japan', 'United Kingdom', 'Canada', 'France']
for country in preferred_defaults:
    if country in available_countries:
        default_countries.append(country)
    if len(default_countries) >= 4:
        break

if not default_countries:
    default_countries = available_countries[:4]

countries = st.sidebar.multiselect("🌍 Select Countries", available_countries, default=default_countries)

variable = st.sidebar.selectbox("📊 Weather Variable", 
                               ['temperature_celsius', 'precip_mm', 'humidity', 'wind_kph'],
                               format_func=lambda x: {
                                   'temperature_celsius': '🌡️ Temperature (°C)',
                                   'precip_mm': '🌧️ Precipitation (mm)',
                                   'humidity': '💧 Humidity (%)',
                                   'wind_kph': '💨 Wind Speed (kph)'
                               }.get(x, x))

year_range = st.sidebar.slider("📅 Year Range", int(df.year.min()), int(df.year.max()), 
                              (int(df.year.min()), int(df.year.max())))
baseline_years = st.sidebar.slider("📊 Baseline Period (for anomaly calculation)", int(df.year.min()), int(df.year.max()), 
                                  (int(df.year.min()), min(int(df.year.min()+9), int(df.year.max()))))

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    show_anomalies = st.checkbox("Show Climate Anomalies", value=True)
    show_trends = st.checkbox("Show Trend Lines", value=True)
    show_extremes = st.checkbox("Highlight Extreme Events", value=True)
    extreme_threshold = st.slider("Extreme Event Threshold (%)", 90, 99, 95)

# Educational mode toggle
education_mode = st.sidebar.toggle("📚 Educational Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Data Export")
if st.sidebar.button("📊 Export Dashboard Data"):
    filtered_export = df[(df.country.isin(countries if countries else available_countries)) & 
                        (df.year.between(*year_range))]
    csv = filtered_export.to_csv(index=False)
    st.sidebar.download_button("📁 Download CSV", csv, "climate_data.csv", "text/csv")

# Display data info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Dataset Information")
st.sidebar.info(f"""
**Total Countries:** {len(available_countries)}  
**Time Period:** {df.year.min()}-{df.year.max()}  
**Total Records:** {len(df):,}  
**Variables:** 4 climate metrics
""")

# Check if countries are selected
if not countries:
    st.warning("⚠️ Please select at least one country from the sidebar to begin exploring!")
    st.info("💡 **Tip:** You can select multiple countries to compare their climate patterns.")
    st.stop()

# Filter data
mask = (df.country.isin(countries)) & (df.year.between(*year_range))
filtered = df[mask]

if filtered.empty:
    st.warning("⚠️ No data available for selected filters. Please adjust your selection.")
    st.info("💡 **Tip:** Try expanding your year range or selecting different countries.")
    st.stop()

# Educational content section
if education_mode:
    edu_content = get_climate_education(variable)
    if edu_content:
        with st.expander(f"📚 Learn About {variable.replace('_', ' ').title()}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🔬 Definition:** {edu_content['definition']}")
                st.markdown(f"**🌍 Why It Matters:** {edu_content['importance']}")
                st.markdown(f"**📊 Normal Range:** {edu_content['normal_range']}")
            with col2:
                st.markdown("**🎯 Key Impacts:**")
                for impact in edu_content['impacts']:
                    st.markdown(f"• {impact}")

                # Show climate tips
                tips = get_climate_tips(variable)
                if tips:
                    st.markdown("**💡 What You Can Do:**")
                    for tip in tips[:2]:  # Show first 2 tips
                        st.markdown(f"• {tip}")

# Calculate anomaly relative to baseline
baseline = df[df.year.between(*baseline_years)].groupby('country')[variable].mean()
filtered['anomaly'] = filtered.apply(lambda r: r[variable] - baseline.get(r.country, 0), axis=1)

# Main dashboard layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Geographic View", "📈 Trends & Analysis", "🔍 Deep Dive", "🎓 Climate Insights", "🌱 Take Action"])

with tab1:
    st.markdown("### 🗺️ Global Climate Visualization")

    # Geographic visualization
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🌍 Interactive World Map")
        map_data = filtered.groupby('country')[variable].mean().reset_index()

        # Choose appropriate color scale
        color_scales = {
            'temperature_celsius': 'RdYlBu_r',
            'precip_mm': 'Blues',
            'humidity': 'Teal',
            'wind_kph': 'Viridis'
        }

        fig_map = px.choropleth(
            map_data, 
            locations='country', 
            locationmode='country names',
            color=variable, 
            color_continuous_scale=color_scales.get(variable, 'Viridis'),
            title=f"Average {variable.replace('_', ' ').title()} ({year_range[0]}-{year_range[1]})",
            hover_data={variable: ':.2f'},
            labels={variable: variable.replace('_', ' ').title()}
        )

        fig_map.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True))
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.subheader("📊 Global Statistics")

        # Calculate key metrics
        global_avg = filtered[variable].mean()
        global_min = filtered[variable].min()
        global_max = filtered[variable].max()
        baseline_avg = baseline[baseline.index.isin(countries)].mean() if not baseline.empty else global_avg

        # Display metrics with context
        st.metric("🌍 Global Average", f"{global_avg:.2f}", f"{global_avg-baseline_avg:.2f}")
        st.metric("📉 Minimum Value", f"{global_min:.2f}")
        st.metric("📈 Maximum Value", f"{global_max:.2f}")

        # Country rankings
        st.subheader("🏆 Country Rankings")
        country_avg = filtered.groupby('country')[variable].mean().sort_values(ascending=False)

        st.markdown("**Highest Values:**")
        for i, (country, value) in enumerate(country_avg.head(3).items()):
            st.write(f"{i+1}. {country}: {value:.2f}")

        # Climate insights
        if education_mode:
            st.subheader("🧠 Smart Insights")
            for country in countries[:3]:  # Limit to 3 countries for space
                country_data = filtered[filtered.country == country]
                if not country_data.empty:
                    insights = get_climate_insights(country_data, variable, country)
                    for insight in insights:
                        st.info(insight)

with tab2:
    st.markdown("### 📈 Climate Trends and Patterns")

    # Time series and trends
    st.subheader("📊 Climate Evolution Over Time")

    fig_line = px.line(
        filtered, 
        x='year_month', 
        y=variable, 
        color='country',
        title=f"{variable.replace('_', ' ').title()} Trends Over Time",
        markers=True,
        labels={variable: variable.replace('_', ' ').title(), 'year_month': 'Time Period'}
    )

    if show_trends:
        # Add trend lines for each country
        for country in countries:
            country_data = filtered[filtered.country == country].groupby('year')[variable].mean().reset_index()
            if len(country_data) > 1:
                slope, intercept = simple_linregress(country_data.year, country_data[variable])
                trend_y = slope * country_data.year + intercept
                fig_line.add_trace(go.Scatter(
                    x=country_data.year.astype(str) + '-01', 
                    y=trend_y,
                    mode='lines',
                    name=f'{country} Trend',
                    line=dict(dash='dash', width=2),
                    opacity=0.8
                ))

    if show_anomalies and 'anomaly' in filtered.columns:
        # Add anomaly indicator
        fig_anomaly = px.scatter(
            filtered[abs(filtered['anomaly']) > filtered['anomaly'].std()], 
            x='year_month', 
            y=variable,
            color='country',
            size=abs(filtered[abs(filtered['anomaly']) > filtered['anomaly'].std()]['anomaly']),
            title="Climate Anomalies (Large Deviations)",
            opacity=0.7
        )
        fig_line.add_traces(fig_anomaly.data)

    fig_line.update_xaxes(rangeslider_visible=True)
    fig_line.update_layout(height=600, hovermode='x unified')
    st.plotly_chart(fig_line, use_container_width=True)

    # Seasonal patterns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔄 Seasonal Patterns")
        seasonal_data = filtered.groupby(['country', 'month'])[variable].mean().reset_index()

        if not seasonal_data.empty:
            fig_seasonal = px.line_polar(
                seasonal_data, 
                r=variable, 
                theta='month',
                color='country',
                line_close=True,
                title="Annual Climate Cycle (Polar View)",
                labels={variable: variable.replace('_', ' ').title()}
            )
            fig_seasonal.update_traces(fill='toself', opacity=0.3)
            fig_seasonal.update_layout(height=400)
            st.plotly_chart(fig_seasonal, use_container_width=True)

    with col2:
        st.subheader("📅 Monthly Heatmap")
        if not seasonal_data.empty:
            seasonal_pivot = seasonal_data.pivot(index='month', columns='country', values=variable)
            fig_heatmap = px.imshow(
                seasonal_pivot,
                title="Monthly Climate Patterns",
                labels={'x': 'Country', 'y': 'Month', 'color': variable.replace('_', ' ').title()},
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.markdown("### 🔍 Advanced Climate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution analysis
        st.subheader("📊 Statistical Distribution")
        fig_box = px.box(
            filtered, 
            x='country', 
            y=variable,
            title=f"{variable.replace('_', ' ').title()} Distribution by Country",
            labels={variable: variable.replace('_', ' ').title()}
        )
        fig_box.update_xaxes(tickangle=45)
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

        # Violin plot for detailed distribution
        fig_violin = px.violin(
            filtered, 
            x='country', 
            y=variable,
            title="Detailed Distribution Shape",
            box=True
        )
        fig_violin.update_xaxes(tickangle=45)
        fig_violin.update_layout(height=400)
        st.plotly_chart(fig_violin, use_container_width=True)

    with col2:
        # Extreme events analysis
        if show_extremes:
            st.subheader("⚡ Extreme Weather Events")

            # Calculate extremes based on user-defined threshold
            threshold_high = filtered[variable].quantile(extreme_threshold/100)
            threshold_low = filtered[variable].quantile((100-extreme_threshold)/100)

            extremes_high = filtered[filtered[variable] >= threshold_high]
            extremes_low = filtered[filtered[variable] <= threshold_low]

            st.metric("🔥 High Extreme Events", len(extremes_high), f"≥{threshold_high:.1f}")
            st.metric("🧊 Low Extreme Events", len(extremes_low), f"≤{threshold_low:.1f}")

            # Show recent extremes
            if not extremes_high.empty:
                st.markdown("**🔥 Recent High Extremes:**")
                extreme_summary = extremes_high.nlargest(5, variable)[['country', 'year_month', variable]]
                st.dataframe(extreme_summary, hide_index=True, use_container_width=True)

            # Extreme events timeline
            if not extremes_high.empty or not extremes_low.empty:
                st.subheader("📈 Extreme Events Timeline")
                fig_extremes = go.Figure()

                if not extremes_high.empty:
                    fig_extremes.add_trace(go.Scatter(
                        x=extremes_high['year_month'],
                        y=extremes_high[variable],
                        mode='markers',
                        name='High Extremes',
                        marker=dict(color='red', size=8),
                        text=extremes_high['country']
                    ))

                if not extremes_low.empty:
                    fig_extremes.add_trace(go.Scatter(
                        x=extremes_low['year_month'],
                        y=extremes_low[variable],
                        mode='markers',
                        name='Low Extremes',
                        marker=dict(color='blue', size=8),
                        text=extremes_low['country']
                    ))

                fig_extremes.update_layout(title="Extreme Weather Events Over Time", height=300)
                st.plotly_chart(fig_extremes, use_container_width=True)

        # Future projection
        st.subheader("🔮 Climate Projection")
        proj_years = st.slider("Project how many years ahead?", 1, 20, 5)
        proj_year = year_range[1] + proj_years

        proj_data = []
        for country in countries:
            country_data = filtered[filtered.country == country]
            if len(country_data) > 2:
                yearly_data = country_data.groupby('year')[variable].mean()
                if len(yearly_data) > 1:
                    slope, intercept = simple_linregress(yearly_data.index, yearly_data.values)
                    proj_val = intercept + slope * proj_year
                    current_val = yearly_data.iloc[-1]
                    change = proj_val - current_val
                    confidence = "High" if len(yearly_data) > 5 else "Medium" if len(yearly_data) > 3 else "Low"
                    proj_data.append({
                        'Country': country,
                        'Current': current_val,
                        'Projected': proj_val,
                        'Change': change,
                        'Confidence': confidence
                    })

        if proj_data:
            proj_df = pd.DataFrame(proj_data)
            fig_proj = px.bar(
                proj_df, 
                x='Country', 
                y=['Current', 'Projected'],
                title=f"Climate Projection for {proj_year}",
                barmode='group',
                color_discrete_map={'Current': 'lightblue', 'Projected': 'orange'}
            )
            fig_proj.update_layout(height=400)
            st.plotly_chart(fig_proj, use_container_width=True)

            # Show projection details
            st.markdown("**📊 Projection Details:**")
            for _, row in proj_df.iterrows():
                change_emoji = "📈" if row['Change'] > 0 else "📉" if row['Change'] < 0 else "➡️"
                st.write(f"{change_emoji} **{row['Country']}:** {row['Change']:+.2f} change ({row['Confidence']} confidence)")

with tab4:
    st.markdown("### 🎓 Climate Science Education")

    if education_mode:
        # Climate change indicators
        st.markdown("### 🌡️ Understanding Climate Change")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📚 Climate Science Facts")
            edu_content = get_climate_education(variable)
            if edu_content and 'facts' in edu_content:
                for fact in edu_content['facts']:
                    st.info(f"💡 {fact}")

        with col2:
            st.markdown("#### 🌍 Global Context")

            # Calculate some educational statistics
            if variable == 'temperature_celsius':
                temp_range = filtered[variable].max() - filtered[variable].min()
                st.success(f"**Temperature Range in Selected Data:** {temp_range:.1f}°C")
                st.warning("🚨 A 2°C global temperature rise is considered the limit to avoid dangerous climate change.")

            elif variable == 'precip_mm':
                precip_variation = filtered[variable].std()
                st.success(f"**Precipitation Variation:** {precip_variation:.1f}mm")
                st.warning("🌊 Climate change is making precipitation patterns more extreme.")

            elif variable == 'humidity':
                humidity_avg = filtered[variable].mean()
                if humidity_avg > 70:
                    st.warning("💧 High humidity regions may become less comfortable due to warming.")
                else:
                    st.info("🏜️ Lower humidity regions may experience more drought stress.")

            elif variable == 'wind_kph':
                wind_potential = "High" if filtered[variable].mean() > 15 else "Medium" if filtered[variable].mean() > 10 else "Low"
                st.success(f"**Wind Energy Potential:** {wind_potential}")
                st.info("💨 Wind patterns are changing due to climate change, affecting renewable energy.")

        # Regional climate patterns education
        st.markdown("### 🗺️ Regional Climate Patterns")

        pattern_insights = []
        for country in countries[:3]:
            country_data = filtered[filtered.country == country]
            if not country_data.empty:
                seasonal_var = country_data.groupby('month')[variable].std().mean()
                total_var = country_data[variable].std()

                if seasonal_var > total_var * 0.6:
                    pattern_insights.append(f"🔄 **{country}** shows strong seasonal variation - typical of continental climates")
                elif seasonal_var < total_var * 0.3:
                    pattern_insights.append(f"🌊 **{country}** has stable year-round patterns - typical of maritime climates")
                else:
                    pattern_insights.append(f"⚖️ **{country}** shows moderate seasonal patterns")

        for insight in pattern_insights:
            st.write(insight)

        # Climate zones explanation
        with st.expander("🌍 Understanding Climate Zones"):
            st.markdown("""
            **Major Climate Types:**

            🏜️ **Arid/Desert:** Very low precipitation, high temperature variation

            🌿 **Temperate:** Moderate temperatures, distinct seasons

            🌴 **Tropical:** High temperatures, high humidity, wet/dry seasons

            ❄️ **Polar:** Very cold temperatures, low precipitation

            🏔️ **Mountain:** Temperature decreases with altitude, varied precipitation

            🌊 **Coastal:** Moderated by ocean, less temperature variation
            """)

with tab5:
    st.markdown("### 🌱 Climate Action & Solutions")

    # Personalized action recommendations
    st.markdown("#### 💪 What You Can Do")

    action_cols = st.columns(3)

    with action_cols[0]:
        st.markdown("##### 🏠 At Home")
        tips = get_climate_tips(variable)
        for tip in tips:
            st.write(f"• {tip}")

        st.markdown("""
        **General Actions:**
        • 💡 Switch to LED bulbs and energy-efficient appliances
        • 🌡️ Use programmable thermostats
        • ♻️ Reduce, reuse, recycle
        • 🚿 Take shorter showers
        """)

    with action_cols[1]:
        st.markdown("##### 🚗 Transportation")
        st.markdown("""
        • 🚶 Walk or bike for short trips
        • 🚌 Use public transportation
        • 🚗 Choose fuel-efficient vehicles
        • ✈️ Reduce air travel when possible
        • 🏠 Work from home when possible
        • 🛒 Combine errands into one trip
        """)

    with action_cols[2]:
        st.markdown("##### 🌱 Lifestyle")
        st.markdown("""
        • 🥗 Eat more plant-based meals
        • 🛒 Buy local and seasonal food
        • 👕 Buy less, choose sustainable brands
        • 🌳 Support reforestation projects
        • 🗳️ Vote for climate-conscious leaders
        • 📚 Learn and share climate knowledge
        """)

    # Community and global actions
    st.markdown("---")
    st.markdown("#### 🌍 Bigger Impact Actions")

    big_impact_tabs = st.tabs(["🏘️ Community", "🏢 Organizations", "🌎 Global"])

    with big_impact_tabs[0]:
        st.markdown("""
        **Community Climate Solutions:**

        🌳 **Urban Greening:** Participate in tree planting programs

        ☀️ **Community Solar:** Join community renewable energy projects

        🚌 **Transportation:** Advocate for better public transport

        🏫 **Education:** Support climate education in schools

        💧 **Water Conservation:** Community rainwater harvesting

        🗳️ **Local Politics:** Engage with local climate policies
        """)

    with big_impact_tabs[1]:
        st.markdown("""
        **Organizational Climate Action:**

        ⚡ **Energy Transition:** Switch to renewable energy

        🏢 **Green Buildings:** Implement sustainable building practices

        📦 **Supply Chain:** Choose sustainable suppliers

        💼 **Remote Work:** Reduce commuting emissions

        🌿 **Carbon Offsetting:** Invest in verified offset programs

        📊 **Measurement:** Track and report carbon footprint
        """)

    with big_impact_tabs[2]:
        st.markdown("""
        **Global Climate Efforts:**

        🌍 **Paris Agreement:** International climate commitments

        💰 **Climate Finance:** Fund developing country adaptations

        🔬 **Technology Innovation:** Invest in clean technology R&D

        🌊 **Nature Conservation:** Protect forests and oceans

        📈 **Carbon Pricing:** Implement carbon taxes and trading

        🤝 **International Cooperation:** Share knowledge and resources
        """)

    # Climate pledge
    st.markdown("---")
    st.markdown("#### ✋ Make a Climate Pledge")

    pledge_options = [
        "🌱 I will reduce my energy consumption at home",
        "🚗 I will use more sustainable transportation",
        "🥗 I will eat more plant-based meals",
        "♻️ I will reduce waste and recycle more",
        "🌳 I will support environmental organizations",
        "📚 I will learn more about climate science",
        "🗳️ I will vote for climate-conscious leaders",
        "💬 I will talk to others about climate action"
    ]

    selected_pledges = []
    for pledge in pledge_options:
        if st.checkbox(pledge):
            selected_pledges.append(pledge)

    if selected_pledges:
        st.success(f"🎉 Thank you for making {len(selected_pledges)} climate pledge(s)!")
        st.balloons()

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 30px;'>

### 📚 About ClimateScope Educational Dashboard

This comprehensive dashboard combines **real climate data** with **educational content** to help you understand global weather patterns and climate change.

**🎯 Purpose:**
- Visualize global climate patterns and trends
- Learn about climate science and meteorology
- Understand the impacts of climate change
- Discover actionable solutions for climate action

**📊 Data Sources:**
- Global Weather Repository Dataset
- Real-time weather station measurements worldwide
- Historical climate records and analysis

**🔗 Learn More:**
- [NASA Climate Change Portal](https://climate.nasa.gov/)
- [NOAA Climate Information](https://www.climate.gov/)
- [IPCC Climate Reports](https://www.ipcc.ch/)
- [Global Carbon Atlas](http://www.globalcarbonatlas.org/)

---
*ClimateScope Dashboard • Built for Climate Education and Awareness*
</div>
""", unsafe_allow_html=True)

# Debug information (collapsible)
with st.expander("🔧 Technical Information & Troubleshooting"):
    st.write(f"**Countries Available:** {len(available_countries)}")
    st.write(f"**Selected Countries:** {countries}")
    st.write(f"**Date Range:** {filtered['year_month'].min()} to {filtered['year_month'].max()}")
    st.write(f"**Total Records:** {len(filtered):,}")
    st.write(f"**Dataset Shape:** {df.shape}")
    st.write(f"**Variables Available:** Temperature, Precipitation, Humidity, Wind Speed")

    if st.button("🔄 Clear Cache and Refresh"):
        st.cache_data.clear()
        st.experimental_rerun()
