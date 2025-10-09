## 修改后的地方：
## （1）去掉description和user guide部分
## （2）对于Employment Rate模块下，去掉Employment Rate Heatmap和Employment Rate Ranking by Field
## (3)对于 Salary Level板块下，去掉 Salary Percentile Comparison by Field这个图
## 总结页的雷达图没有去掉旋转的功能，因为default就是会旋转

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re ##新的
# Page configuration
st.set_page_config(
    page_title="Singapore University Graduate Employment Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
########新的#######
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
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


#############################################################################
# === new: 五大类映射与工具函数 ===
five_groups = ["Business", "Engineering", "Arts & Social Sciences", "Science", "Law"]

patterns = [
    (r'\blaw\b|\bjd\b', 'Law'),
    (r'\bengineer\w*\b|\binformation systems\b|\bcomput\w*\b', 'Engineering'),
    (r'\bbusiness\b|\baccountancy\b|\bfinance\b|\bmarketing\b', 'Business'),
    (r'\bsocial\b|\beconom(ics|y)\b|\bhumanit\w*\b|\barts?\b', 'Arts & Social Sciences'),
    (r'\bscience\b|\bmath(ematics)?\b|\bphysics\b|\bchemistry\b|\bbiology\b', 'Science'),
]

def classify_to_five_groups(text: str):
    """根据院系名称匹配五大类"""
    t = str(text).lower()
    for pat, label in patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            return label
    return None

# 生成分组列
if "school_group" not in df.columns:
    if "school" in df.columns:
        df["school_group"] = df["school"].apply(classify_to_five_groups)
    else:
        df["school_group"] = df["field"].apply(classify_to_five_groups)

# 剔除无法对比的学科
exclude_keys = ["medicine", "dentistry", "design", "environment"]
df = df[~df.apply(lambda r: any(k in str(r.get("school", r.get("field", ""))).lower() for k in exclude_keys), axis=1)]

# 工资列优先使用 basic
salary_col = "basic_monthly_median" 

####################################################################################

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
    options=["Employment Rate", 
             "Salary Level", 
             "Employment Stability", 
             "Five Field Groups",##新加的
             "Comprehensive Comparison"]
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

# Check if filtered data is available
if len(filtered_df) == 0:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Display different analysis content based on selected metric
if metric == "Employment Rate":
    st.header("📊 Employment Rate Analysis")
    
    # Employment rate trends by field (full width)
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
    
    # Salary growth rate (full width)
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

# ========== ✅ 新增页面：Five Field Groups（精简版：仅箱线图） ==========
elif metric == "Five Field Groups":
    st.header("🏷️ Five Field Groups")

    # —— 本页独立筛选（不沿用主侧栏的 field） ——
    with st.container():
        c1, c2 = st.columns([1.2, 1.2])
        with c1:
            view_choice = st.radio(
                "Choose Metric ",
                ("Employment Rate", "Salary Level"),
                horizontal=True
            )
        with c2:
            groups_selected = st.multiselect(
                "Choose Groups（默认全选）",
                options=five_groups,
                default=five_groups
            )

    metric_col = (
        "employment_rate_overall"
        if view_choice == "Employment Rate"
        else salary_col
    )

    if view_choice == "Employment Rate":
        y_label = "Employment Rate"
    else:
        y_label = "Basic Monthly Median (SGD)"


    # 仅用"年份 + 大学"过滤；不按 field 过滤，改用本页分组
    base = (
        df[
            df["year"].between(selected_years[0], selected_years[1])
            & df["university"].isin(universities)
            & df["school_group"].isin(groups_selected)
        ]
        .dropna(subset=["school_group", "university", metric_col])
    )

    if base.empty:
        st.info("There is no data available for the current filter. Please adjust the year or university.")
        st.stop()

    st.markdown("---")
    st.subheader(f"{view_choice} by University")
    st.caption("说明：每个分组面板展示该分组内各大学的分布（箱线：Q1~Q3，线：中位数，点：离群值）。")

    # —— 每个分组一张小面板：箱线图（大学为类别，纵轴为所选指标） ——
    cols = st.columns(2)
    idx = 0
    for grp in groups_selected:
        d = base.loc[base["school_group"] == grp, ["university", "year", metric_col]].copy()
        if d.empty:
            continue

        # 根据"各大学的中位数"排序横轴，读图更直观
        univ= {
            "National University of Singapore": "NUS",
            "Nanyang Technological University": "NTU",
            "Singapore Management University": "SMU",
            "Singapore University of Social Sciences": "SUSS",
        }
        d["university_univ"] = d["university"].replace(univ)

        order = (
            d.groupby("university_univ")[metric_col]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig = px.box(
            d,
            x="university_univ",
            y=metric_col,
            points="outliers",  # 显示离群点；如不需要可改为 False
            category_orders={"university_univ": order},
            title=f"{grp} — {y_label} by University",
            labels={"university_univ": "University", metric_col: y_label},
            hover_data=["year"]
        )

        if view_choice == "Employment Rate":           
            fig.update_layout(yaxis_tickformat=".0%")
        else:
            fig.update_yaxes(tickprefix="$", separatethousands=True)

        fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))

        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)
        idx += 1
    # ======= B. Summary Table: Highest-performing Universities per Group =======
    st.markdown("---")
    st.subheader(f"Summary: Top Universities by {view_choice}")
    summary = (
    base.groupby(["school_group", "university"], as_index=False)[metric_col]
        .mean()
        .sort_values(["school_group", metric_col], ascending=[True, False])
    )

    # 每个组取最高的学校
    top_summary = summary.groupby("school_group").head(1).copy()

    # 格式化数值显示
    if view_choice == "Employment Rate":
        top_summary[metric_col] = top_summary[metric_col].apply(lambda x: f"{x:.1%}")
    else:
        top_summary[metric_col] = top_summary[metric_col].apply(lambda x: f"${x:,.0f}")

    # 列名友好化
    top_summary = top_summary.rename(columns={
        "school_group": "Group",
        "university": "Top University",
        metric_col: view_choice
    })
    top_summary = top_summary.reset_index(drop=True)
    top_summary.index = top_summary.index + 1

    # 展示表格
    st.dataframe(
        top_summary,
        use_container_width=True,
        height=min(400, 80 + 30 * len(top_summary))
    )

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
