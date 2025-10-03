import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Singapore University Graduate Employment Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data from your Excel file
@st.cache_data
def load_data():
    # Load data from your specific file path
    file_path = "data_universities.xlsx"
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name='classified_universities')
        
        # Data cleaning and preprocessing
        # Ensure numeric columns are properly formatted
        numeric_columns = [
            'employment_rate_overall', 'employment_rate_ft_perm',
            'basic_monthly_mean', 'basic_monthly_median',
            'gross_monthly_mean', 'gross_monthly_median',
            'gross_mthly_25_percentile', 'gross_mthly_75_percentile'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['field', 'year', 'employment_rate_overall', 'gross_monthly_median'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty dataframe as fallback
        return pd.DataFrame()

# Load data
df = load_data()

# Check if data is loaded successfully
if df.empty:
    st.error("No data loaded. Please check the file path and format.")
    st.stop()

# Sidebar
st.sidebar.title("🎓 Analysis Settings")

# Year selection
years = sorted(df['year'].unique())
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min(years),
    max_value=max(years),
    value=(min(years), max(years))
)

# Field selection
available_fields = sorted(df['field'].unique())
fields = st.sidebar.multiselect(
    "Select Fields of Study",
    options=available_fields,
    default=available_fields
)

# University selection
available_universities = sorted(df['university'].unique())
universities = st.sidebar.multiselect(
    "Select Universities",
    options=available_universities,
    default=available_universities
)

# Metric selection
metric = st.sidebar.selectbox(
    "Select Analysis Metric",
    options=["Employment Rate", "Salary Level", "Employment Stability", "Comprehensive Comparison"]
)

# Filter data based on selections
filtered_df = df[
    (df['year'].between(selected_years[0], selected_years[1])) &
    (df['field'].isin(fields)) &
    (df['university'].isin(universities))
]

# Main page
st.title("🎓 Singapore University Graduate Employment Analysis Dashboard")
st.markdown("---")

# Display data overview
st.sidebar.markdown("---")
st.sidebar.subheader("Data Overview")
st.sidebar.write(f"Total Records: {len(df):,}")
st.sidebar.write(f"Filtered Records: {len(filtered_df):,}")
st.sidebar.write(f"Years: {df['year'].min()} - {df['year'].max()}")
st.sidebar.write(f"Fields: {len(available_fields)}")
st.sidebar.write(f"Universities: {len(available_universities)}")

# Key metrics cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_employment = filtered_df['employment_rate_overall'].mean()
    st.metric(
        "Average Employment Rate", 
        f"{avg_employment:.1%}",
        delta=f"{(avg_employment - df['employment_rate_overall'].mean()):.2%}" if len(filtered_df) > 0 else "0%"
    )

with col2:
    avg_salary = filtered_df['gross_monthly_median'].mean()
    st.metric(
        "Median Monthly Salary", 
        f"${avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A",
        delta=f"${avg_salary - df['gross_monthly_median'].mean():.0f}" if len(filtered_df) > 0 and not pd.isna(avg_salary) else "0"
    )

with col3:
    ft_employment = filtered_df['employment_rate_ft_perm'].mean()
    st.metric(
        "Full-time Employment Rate", 
        f"{ft_employment:.1%}" if not pd.isna(ft_employment) else "N/A",
        delta=f"{(ft_employment - df['employment_rate_ft_perm'].mean()):.2%}" if len(filtered_df) > 0 and not pd.isna(ft_employment) else "0%"
    )

with col4:
    if len(filtered_df) > 0:
        salary_gap = filtered_df['gross_mthly_75_percentile'].mean() - filtered_df['gross_mthly_25_percentile'].mean()
        st.metric(
            "Salary Range", 
            f"${salary_gap:,.0f}" if not pd.isna(salary_gap) else "N/A",
            delta="Salary distribution range"
        )
    else:
        st.metric("Salary Range", "N/A")

st.markdown("---")

# Check if filtered data is available
if len(filtered_df) == 0:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Display different analysis content based on selected metric
if metric == "Employment Rate":
    st.header("📊 Employment Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Employment rate trends by field
        yearly_employment = filtered_df.groupby(['year', 'field'])['employment_rate_overall'].mean().reset_index()
        fig1 = px.line(
            yearly_employment, 
            x='year', 
            y='employment_rate_overall', 
            color='field',
            title="Employment Rate Trends by Field",
            labels={'employment_rate_overall': 'Employment Rate', 'year': 'Year', 'field': 'Field of Study'}
        )
        fig1.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Current employment rate ranking
        current_employment = filtered_df[filtered_df['year'] == selected_years[1]].groupby('field')['employment_rate_overall'].mean().sort_values(ascending=False)
        fig2 = px.bar(
            x=current_employment.values,
            y=current_employment.index,
            orientation='h',
            title=f"Employment Rate Ranking by Field ({selected_years[1]})",
            labels={'x': 'Employment Rate', 'y': 'Field of Study'}
        )
        fig2.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Employment rate heatmap
    st.subheader("Employment Rate Heatmap")
    heatmap_data = filtered_df.pivot_table(
        index='field', 
        columns='year', 
        values='employment_rate_overall', 
        aggfunc='mean'
    ).fillna(0)
    fig3 = px.imshow(
        heatmap_data,
        title="Employment Rate Heatmap by Field and Year",
        color_continuous_scale='Blues',
        aspect="auto"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # University comparison by field
    st.subheader("University Comparison by Field")
    university_field_employment = filtered_df.groupby(['university', 'field'])['employment_rate_overall'].mean().reset_index()
    fig4 = px.box(
        university_field_employment,
        x='field',
        y='employment_rate_overall',
        color='university',
        title="Employment Rate Distribution by University and Field",
        labels={'employment_rate_overall': 'Employment Rate', 'field': 'Field of Study', 'university': 'University'}
    )
    fig4.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)

elif metric == "Salary Level":
    st.header("💰 Salary Level Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary trends by field
        yearly_salary = filtered_df.groupby(['year', 'field'])['gross_monthly_median'].mean().reset_index()
        fig1 = px.line(
            yearly_salary, 
            x='year', 
            y='gross_monthly_median', 
            color='field',
            title="Median Monthly Salary Trends by Field",
            labels={'gross_monthly_median': 'Median Monthly Salary (SGD)', 'year': 'Year', 'field': 'Field of Study'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Salary distribution box plot
        fig2 = px.box(
            filtered_df, 
            x='field', 
            y='gross_monthly_median',
            title="Salary Distribution by Field",
            labels={'gross_monthly_median': 'Median Monthly Salary (SGD)', 'field': 'Field of Study'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Salary growth rate
        try:
            salary_growth = filtered_df.groupby('field').apply(
                lambda x: (x[x['year'] == selected_years[1]]['gross_monthly_median'].mean() - 
                          x[x['year'] == selected_years[0]]['gross_monthly_median'].mean()) / 
                         x[x['year'] == selected_years[0]]['gross_monthly_median'].mean()
            ).sort_values(ascending=False)
            
            fig3 = px.bar(
                x=salary_growth.values,
                y=salary_growth.index,
                orientation='h',
                title=f"Salary Growth Rate ({selected_years[0]}-{selected_years[1]})",
                labels={'x': 'Growth Rate', 'y': 'Field of Study'}
            )
            fig3.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)
        except:
            st.info("Cannot calculate salary growth with current filter selection.")
    
    with col4:
        # Salary percentile comparison
        salary_stats = filtered_df.groupby('field').agg({
            'gross_mthly_25_percentile': 'mean',
            'gross_monthly_median': 'mean',
            'gross_mthly_75_percentile': 'mean'
        }).reset_index()
        
        # Create a melted dataframe for easier plotting
        salary_melted = salary_stats.melt(id_vars=['field'], 
                                         value_vars=['gross_mthly_25_percentile', 'gross_monthly_median', 'gross_mthly_75_percentile'],
                                         var_name='percentile', 
                                         value_name='salary')
        
        # Map percentile names to more readable format
        percentile_map = {
            'gross_mthly_25_percentile': '25th Percentile',
            'gross_monthly_median': 'Median',
            'gross_mthly_75_percentile': '75th Percentile'
        }
        salary_melted['percentile'] = salary_melted['percentile'].map(percentile_map)
        
        fig4 = px.bar(
            salary_melted,
            x='field',
            y='salary',
            color='percentile',
            barmode='group',
            title="Salary Percentile Comparison by Field",
            labels={'salary': 'Monthly Salary (SGD)', 'field': 'Field of Study', 'percentile': 'Percentile'}
        )
        st.plotly_chart(fig4, use_container_width=True)

elif metric == "Employment Stability":
    st.header("📈 Employment Stability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Full-time employment rate comparison
        ft_employment = filtered_df.groupby('field')['employment_rate_ft_perm'].mean().sort_values(ascending=False)
        fig1 = px.bar(
            x=ft_employment.values,
            y=ft_employment.index,
            orientation='h',
            title="Full-time Employment Rate by Field",
            labels={'x': 'Full-time Employment Rate', 'y': 'Field of Study'}
        )
        fig1.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Employment stability trend
        stability_trend = filtered_df.groupby(['year', 'field']).agg({
            'employment_rate_overall': 'mean',
            'employment_rate_ft_perm': 'mean'
        }).reset_index()
        stability_trend['stability_gap'] = stability_trend['employment_rate_overall'] - stability_trend['employment_rate_ft_perm']
        
        fig2 = px.line(
            stability_trend, 
            x='year', 
            y='stability_gap', 
            color='field',
            title="Employment Stability Gap Trend (Overall - Full-time Employment Rate)",
            labels={'stability_gap': 'Employment Stability Gap', 'year': 'Year', 'field': 'Field of Study'}
        )
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Employment rate vs salary scatter plot
    st.subheader("Employment Rate vs Salary Relationship")
    scatter_data = filtered_df.groupby('field').agg({
        'employment_rate_overall': 'mean',
        'gross_monthly_median': 'mean',
        'employment_rate_ft_perm': 'mean'
    }).reset_index()
    
    fig3 = px.scatter(
        scatter_data,
        x='employment_rate_overall',
        y='gross_monthly_median',
        size='employment_rate_ft_perm',
        color='field',
        hover_name='field',
        title="Employment Rate vs Salary Bubble Chart",
        labels={
            'employment_rate_overall': 'Average Employment Rate',
            'gross_monthly_median': 'Average Median Monthly Salary (SGD)',
            'employment_rate_ft_perm': 'Full-time Employment Rate'
        },
        size_max=60
    )
    fig3.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)
    
    # University performance by field
    st.subheader("University Performance by Field")
    university_performance = filtered_df.groupby(['university', 'field']).agg({
        'employment_rate_overall': 'mean',
        'gross_monthly_median': 'mean'
    }).reset_index()
    
    fig4 = px.scatter(
        university_performance,
        x='employment_rate_overall',
        y='gross_monthly_median',
        color='field',
        symbol='university',
        title="University Performance: Employment Rate vs Salary by Field",
        labels={
            'employment_rate_overall': 'Employment Rate',
            'gross_monthly_median': 'Median Monthly Salary (SGD)',
            'field': 'Field of Study',
            'university': 'University'
        },
        hover_name='university'
    )
    fig4.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)

else:  # Comprehensive Comparison
    st.header("🔍 Comprehensive Comparison Analysis")
    
    # Radar chart - multi-dimensional comparison
    st.subheader("Multi-dimensional Field Comparison")
    
    # Calculate comprehensive metrics for each field
    radar_metrics = filtered_df.groupby('field').agg({
        'employment_rate_overall': 'mean',
        'employment_rate_ft_perm': 'mean',
        'gross_monthly_median': 'mean',
        'gross_mthly_75_percentile': 'mean'
    }).reset_index()
    
    # Normalize data (0-1)
    for col in ['employment_rate_overall', 'employment_rate_ft_perm', 'gross_monthly_median', 'gross_mthly_75_percentile']:
        radar_metrics[f'{col}_normalized'] = (
            radar_metrics[col] - radar_metrics[col].min()
        ) / (radar_metrics[col].max() - radar_metrics[col].min())
    
    # Create radar chart
    fig_radar = go.Figure()
    
    for idx, field in enumerate(radar_metrics['field']):
        field_data = radar_metrics[radar_metrics['field'] == field].iloc[0]
        values = [
            field_data['employment_rate_overall_normalized'],
            field_data['employment_rate_ft_perm_normalized'],
            field_data['gross_monthly_median_normalized'],
            field_data['gross_mthly_75_percentile_normalized']
        ]
        # Close the radar chart
        values.append(values[0])
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=['Employment Rate', 'Full-time Rate', 'Median Salary', 'High Salary Potential', 'Employment Rate'],
            fill='toself',
            name=field
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Comprehensive Field Comparison Radar Chart"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Detailed data table
    st.subheader("Detailed Data Table")
    
    summary_table = filtered_df.groupby('field').agg({
        'employment_rate_overall': ['mean', 'std'],
        'employment_rate_ft_perm': ['mean', 'std'],
        'gross_monthly_median': ['mean', 'std'],
        'gross_monthly_mean': ['mean', 'std']
    }).round(3)
    
    # Rename columns
    summary_table.columns = ['Employment Rate_Mean', 'Employment Rate_Std', 
                            'Full-time Rate_Mean', 'Full-time Rate_Std',
                            'Median Salary_Mean', 'Median Salary_Std',
                            'Mean Salary_Mean', 'Mean Salary_Std']
    
    st.dataframe(summary_table, use_container_width=True)
    
    # Field progression over years
    st.subheader("Field Progression Over Years")
    progression_data = filtered_df.groupby(['year', 'field']).agg({
        'employment_rate_overall': 'mean',
        'gross_monthly_median': 'mean'
    }).reset_index()
    
    # Create a line chart for employment rate progression
    fig_progression1 = px.line(
        progression_data,
        x='year',
        y='employment_rate_overall',
        color='field',
        title="Employment Rate Progression by Field",
        labels={'employment_rate_overall': 'Employment Rate', 'year': 'Year', 'field': 'Field of Study'}
    )
    fig_progression1.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_progression1, use_container_width=True)
    
    # Create a line chart for salary progression
    fig_progression2 = px.line(
        progression_data,
        x='year',
        y='gross_monthly_median',
        color='field',
        title="Salary Progression by Field",
        labels={'gross_monthly_median': 'Median Monthly Salary (SGD)', 'year': 'Year', 'field': 'Field of Study'}
    )
    st.plotly_chart(fig_progression2, use_container_width=True)

# Data description
st.markdown("---")
with st.expander("📋 Data Description"):
    st.write("""
    **Data Field Description:**
    - **Employment Rate**: Proportion of graduates employed after graduation
    - **Full-time Employment Rate**: Proportion of graduates in full-time permanent positions
    - **Median Monthly Salary**: Median monthly salary of graduates
    - **Mean Monthly Salary**: Average monthly salary of graduates
    - **25th Percentile Salary**: 25th percentile of salary distribution
    - **75th Percentile Salary**: 75th percentile of salary distribution
    
    **Field of Study Description:**
    - **Engineering**: Includes electrical, mechanical, civil engineering, etc.
    - **Computing**: Computer science, information technology, and related fields
    - **Business**: Business management, accounting, finance, etc.
    - **Science**: Physics, chemistry, biology, and other natural sciences
    - **Social Sciences**: Sociology, psychology, economics, etc.
    - **Law**: Legal studies
    - **Medicine**: Medical and dental studies
    """)

# Usage guide
with st.expander("🚀 User Guide"):
    st.write("""
    **How to Use This Dashboard:**
    
    1. **Filter Data**: Use the left sidebar to select year range, fields of study, and universities
    2. **Select Analysis Dimension**: Choose the metric type you want to focus on
    3. **Interactive Exploration**: 
       - Hover over charts to see detailed data
       - Click on legend items to show/hide specific data series
       - Use chart toolbar for zooming, panning, and other actions
    4. **Multi-dimensional Analysis**: Combine different charts for comprehensive insights
    
    **Analysis Recommendations:**
    - Focus on long-term trends rather than single-year data
    - Combine employment rate and salary data for balanced judgment
    - Consider the development characteristics and employment stability of different fields
    - Consider personal interests alongside market demand
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎓 Singapore University Graduate Employment Analysis Dashboard | "
    "Data Source: Singapore Ministry of Education | "
    "Last Updated: 2024"
    "</div>",
    unsafe_allow_html=True
)
