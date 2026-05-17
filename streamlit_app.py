import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_theme(style="whitegrid")

##Header
st.set_page_config(page_title="Madrid 10K Performance Hub", layout="wide")

st.title("🏃‍♂️ San Silvestre Vallecana 2019 Interactive Dashboard")
st.markdown("""
### Purpose and Importance of the Application
This application serves as an interactive data analysis toolkit built from the 2019 Madrid 10k race dataset. 
By translating raw split coordinates into dynamic, user-controlled charts, this platform allows coaches, sports analysts, 
and marathon event coordinators to investigate execution pacing baselines across varying age brackets and gender demographics.
""")

st.write("---")

# ==========================================
# 2. DATA LOAD & CACHED EDA PIPELINE (EDIT THIS EDA FROM AI)
# ==========================================
@st.cache_data
def load_clean_data():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_directory, 'madrid_10k_20191231.csv')
    
    df = pd.read_csv(csv_path)
    
    # Clean records missing critical milestone points
    df = df.dropna(subset=['total_seconds', '5km_seconds', 'age_category', 'sex'])
    df = df.drop_duplicates()
    
    # Feature Engineering: Convert raw metric seconds to operational minutes
    df['Total_Minutes'] = df['total_seconds'] / 60
    df['Split_2.5K_Minutes'] = df['2.5km_seconds'] / 60
    df['Split_5K_Minutes'] = df['5km_seconds'] / 60
    df['Split_7.5K_Minutes'] = df['7.5km_seconds'] / 60
    
    df['sex'] = df['sex'].str.upper().str.strip()
    return df

try:
    df = load_clean_data()
except Exception as e:
    st.error(f"⚠️ Dataset mapping error: {e}")
    st.info("Ensure 'madrid_10k_20191231.csv' is right next to this app.py file inside your Downloads folder.")
    st.stop()

##Sidebar
st.sidebar.header("🎯 Live Filter Controls")
st.sidebar.markdown("Use these adjustments to segment the race field:")

#Sb_Gender
gender_list = sorted(list(df['sex'].unique()))
selected_genders = st.sidebar.multiselect("Select Gender Group:", gender_list, default=gender_list)

#Sb_Age
age_categories = sorted([cat for cat in df['age_category'].dropna().unique()])
selected_ages = st.sidebar.multiselect("Select Age Categories:", age_categories, default=age_categories)

#Sb_Time
min_min = float(np.floor(df['Total_Minutes'].min()))
max_min = float(np.ceil(df['Total_Minutes'].max()))
selected_time_range = st.sidebar.slider(
    "Set Completion Time Window (Minutes):",
    min_value=min_min,
    max_value=max_min,
    value=(min_min, max_min),
    step=0.5
)

filtered_df = df[
    (df['sex'].isin(selected_genders)) & 
    (df['age_category'].isin(selected_ages)) & 
    (df['Total_Minutes'] >= selected_time_range[0]) & 
    (df['Total_Minutes'] <= selected_time_range[1])
]

# Safeguard check if user isolates down to an empty collection
if filtered_df.empty:
    st.warning("⚠️ No data items match your exact selected filter targets. Adjust the sidebar sliders or choose more groups!")
    st.stop()

# ==========================================
# 4. FIVE DATA FUNCTIONALITIES (Req. c & d)
# ==========================================

##Overview of Data
st.subheader("📊 Functionality 1: Live Segment Overview & Data Matrix")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Competitors in View", f"{filtered_df.shape[0]:,}")
col_m2.metric("Mean Completion Pace", f"{filtered_df['Total_Minutes'].mean():.1f} mins")
col_m3.metric("Mean Halfway Split (5K)", f"{filtered_df['Split_5K_Minutes'].mean():.1f} mins")
col_m4.metric("Fastest Clock Time", f"{filtered_df['Total_Minutes'].min():.1f} mins")

##Raw Data 
with st.expander("🔎 Click to expand raw spreadsheet records for this selection"):
    st.dataframe(filtered_df[['id_number', 'place', 'age_category', 'sex', 'Split_5K_Minutes', 'Total_Minutes']].head(100), use_container_width=True)

st.write("---")

# Layout Column Grid Split
col1, col2 = st.columns(2)

with col1:
    ##Histogram
    st.subheader("⏱️ Functionality 2: Finish Time Frequency Density")
    
    bins_slider = st.slider("Adjust Histogram Bar Granularity (Bins):", min_value=10, max_value=100, value=30)
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(data=filtered_df, x='Total_Minutes', kde=True, color='skyblue', bins=bins_slider, ax=ax1)
    ax1.axvline(filtered_df['Total_Minutes'].mean(), color='red', linestyle='--', label=f"Mean: {filtered_df['Total_Minutes'].mean():.1f}m")
    ax1.set_title('Overall 10K Completion Spectrum')
    ax1.set_xlabel('Total Duration (Minutes)')
    ax1.set_ylabel('Total Registrants Count')
    ax1.legend()
    st.pyplot(fig1)

with col2:
    ##Demograph: Age & Gender
    st.subheader("🍰 Functionality 3: Demographic Concentration Toggles")
    
    chart_toggle = st.radio("Switch visual view to:", ["Age Division Spread", "Gender Proportions"])
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if chart_toggle == "Age Division Spread":
        sns.countplot(data=filtered_df, x='age_category', order=sorted(filtered_df['age_category'].unique()), palette='viridis', ax=ax2)
        ax2.set_title('Runner Densities Across Age Brackets')
        ax2.set_xlabel('Age Category Class')
    else:
        sns.countplot(data=filtered_df, x='sex', palette='Set2', ax=ax2)
        ax2.set_title('Runner Densities by Gender Category')
        ax2.set_xlabel('Sex Group')
        
    ax2.set_ylabel('Count')
    st.pyplot(fig2)

st.write("---")
col3, col4 = st.columns(2)

with col3:
    ##Quartiles
    st.subheader("📉 Functionality 4: Multi-Category Pacing Interquartiles")
    
    fig3, ax3 = plt.subplots(figsize=(6, 4.3))
    sns.boxplot(data=filtered_df, x='age_category', y='Total_Minutes', hue='sex', 
                order=sorted(filtered_df['age_category'].unique()), palette='Pastel1', ax=ax3)
    ax3.set_title('Race Duration Quartiles Across Demographics')
    ax3.set_xlabel('Age Divisions')
    ax3.set_ylabel('Total Finish Time (Minutes)')
    ax3.legend(title='Gender')
    st.pyplot(fig3)

with col4:
    ##Finish time evaluation
    st.subheader("🎯 Functionality 5: Empirical Field Performance Calculator")
    st.markdown("Enter an individual runner's finish mark below to evaluate how they match up against this chosen field:")
    
    user_mins = st.number_input(
        "Enter Target 10K Finish Time (Minutes):", 
        min_value=20.0, max_value=140.0, value=55.0, step=0.5
    )
    
    # Dynamic backend scientific calculation 
    total_count = filtered_df.shape[0]
    slower_count = filtered_df[filtered_df['Total_Minutes'] > user_mins].shape[0]
    percentile_score = (slower_count / total_count) * 100
    
    st.success(f"🏆 **Evaluation:** A time of **{user_mins:.1f} minutes** performs faster than **{percentile_score:.1f}%** of the currently filtered runners pool!")

st.write("---")

##Correlation Matrix
st.subheader("🌡️ Functional Extra: Correlation Pacing Map Selector")
col5, col6 = st.columns([1.2, 1])

with col5:
    chosen_metric = st.selectbox(
        "Select intermediate benchmark column to correlate with total finish time:",
        ['Split_2.5K_Minutes', 'Split_5K_Minutes', 'Split_7.5K_Minutes']
    )
    
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    corr_data = filtered_df[[chosen_metric, 'Total_Minutes']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".4f", square=True, cbar=True, ax=ax4)
    ax4.set_title(f'Correlation Matrix: {chosen_metric} vs Total Minutes')
    st.pyplot(fig4)

with col6:
    st.markdown(f"""
    **Interactive Correlation Insight:**
    * You are currently inspecting the interactive link between **{chosen_metric}** and final finish time.
    * A coefficient near **1.0000** indicates that performance at that specific checkpoint strongly anchors your eventual placement. 
    * Changing the dropdown allows you to witness mathematically how pacing correlation strengthens or stabilizes as athletes approach the final 10K marker finish line.
    """)
