import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import TEST_STRESS_EVENTS_PATH, TEST_MONTHLY_SUMMARY_PATH, TEST_TOP_EVENTS_PATH
from src.aggregate_results import run_aggregation

#Page Setup
st.set_page_config(
    page_title="Responsible AI for Energy Poverty Detection in Ontario",
    page_icon="⚡",
    layout="wide"
)

st.title("Responsible AI for Energy Poverty Detection in Ontario")
st.markdown(
    """
    This dashboard presents a PoC early-warning system for detecting
    **community energy stress in Ontario** using **public electricity demand data**
    and **weather context**.  
    The system is privacy-preserving, uses no household-level data, and is designed
    for human-in-the-loop decision support.
    """
)

def ensure_outputs_exist():
    files_exist = all([
        os.path.exists(TEST_STRESS_EVENTS_PATH),
        os.path.exists(TEST_MONTHLY_SUMMARY_PATH),
        os.path.exists(TEST_TOP_EVENTS_PATH),
    ])

    if not files_exist:
        st.info("Phase 5 outputs not found. Running aggregation pipeline...")
        run_aggregation(
            anomaly_results_dir="outputs/anomaly_results",
            output_dir="outputs/aggregated_results"
        )

#Helpers
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "datetime" in result.columns:
        result["datetime"] = pd.to_datetime(result["datetime"])
        result["date"] = result["datetime"].dt.date
    return result

def load_dashboard_data():
    ensure_outputs_exist()
    
    test_stress_df = load_csv(TEST_STRESS_EVENTS_PATH)
    monthly_summary_df = load_csv(TEST_MONTHLY_SUMMARY_PATH)
    top_events_df = load_csv(TEST_TOP_EVENTS_PATH)
    
    test_stress_df = add_datetime_columns(test_stress_df)
    
    return test_stress_df, monthly_summary_df, top_events_df

def safe_count(df: pd.DataFrame, col: str, value=1) -> int:
    if col not in df.columns:
        return 0
    return int((df[col] == value).sum())

#Load Data
try:
    test_stress_df, monthly_summary_df, top_events_df = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading dashboard data: {e}")
    st.stop()
    
#Filters in sidebar
st.sidebar.header("Filters")

min_date = test_stress_df["datetime"].min().date()
max_date = test_stress_df["datetime"].max().date()

selected_date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

show_anomalies_only = st.sidebar.checkbox("Show anomalies only", value=False)
show_stress_events_only = st.sidebar.checkbox("Show stress events only", value=False)

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date, end_date = min_date, max_date
    
filtered_df = test_stress_df[
    (test_stress_df["datetime"].dt.date >= start_date) &
    (test_stress_df["datetime"].dt.date <= end_date)
].copy()

if show_anomalies_only and "anomaly_flag" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["anomaly_flag"] == 1]

if show_stress_events_only and "stress_event" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["stress_event"] == 1]
    
#KPI Section
st.subheader("Early Warning Summary")

total_rows = len(filtered_df)
total_anomalies = safe_count(filtered_df, "anomaly_flag", 1)
total_stress_events = safe_count(filtered_df, "stress_event", 1)
cold_stress_events = int(((filtered_df.get("stress_event", 0) == 1) & (filtered_df.get("cold_stress", 0) == 1)).sum()) if "stress_event" in filtered_df.columns and "cold_stress" in filtered_df.columns else 0
heat_stress_events = int(((filtered_df.get("stress_event", 0) == 1) & (filtered_df.get("heat_stress", 0) == 1)).sum()) if "stress_event" in filtered_df.columns and "heat_stress" in filtered_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", f"{total_rows:,}")
col2.metric("Anomalies", f"{total_anomalies:,}")
col3.metric("Stress Events", f"{total_stress_events:,}")
col4.metric("Cold / Heat Events", f"{cold_stress_events:,} / {heat_stress_events:,}")

#Tabs
tab1, tab2, tab3 = st.tabs([
    "Demand & Anomalies",
    "Weather Stress Context",
    "Early Warning Summary"
])

#Tab 1 - Anomalies & Demand
with tab1:
    st.subheader("Ontario Demand and Detected Anomalies")
    
    if "hourly_demand" not in filtered_df.columns:
        st.warning("Column 'hourly_demand' not found.")
    else:
        fig_demand = px.line(
            filtered_df,
            x="datetime",
            y="hourly_demand",
            title="Hourly Ontario Electricity Demand",
            labels={"datetime": "Datetime", "hourly_demand": "Hourly Demand"}
        )
        st.plotly_chart(fig_demand, use_container_width=True)
        
        if "anomaly_flag" in filtered_df.columns:
            anomaly_points = filtered_df[filtered_df["anomaly_flag"] == 1].copy()
            
            if not anomaly_points.empty:
                fig_anomalies = px.scatter(
                    anomaly_points,
                    x="datetime",
                    y="hourly_demand",
                    color="reconstruction_error" if "reconstruction_error" in anomaly_points.columns else None,
                    title="Detected Anomaly Points",
                    labels={
                        "datetime": "Datetime",
                        "hourly_demand": "Hourly Demand",
                        "reconstruction_error": "Reconstruction Error"
                    }
                )
                st.plotly_chart(fig_anomalies, use_container_width=True)
            else:
                st.info("No anomaly points found in the selected range.")
                
#Tab 2: Weather Stress context
with tab2:
    st.subheader("Weather Stress Context")

    if "ontario_avg_temp_c" not in filtered_df.columns:
        st.warning("Column 'ontario_avg_temp_c' not found.")
    else:
        fig_temp = px.line(
            filtered_df,
            x="datetime",
            y="ontario_avg_temp_c",
            title="Ontario Average Temperature",
            labels={"datetime": "Datetime", "ontario_avg_temp_c": "Avg Temp (°C)"}
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        if "stress_event" in filtered_df.columns:
            stress_points = filtered_df[filtered_df["stress_event"] == 1].copy()

            if not stress_points.empty:
                fig_stress = px.scatter(
                    stress_points,
                    x="datetime",
                    y="ontario_avg_temp_c",
                    color="temp_stress_type" if "temp_stress_type" in stress_points.columns else None,
                    symbol="severity" if "severity" in stress_points.columns else None,
                    title="Weather-Linked Stress Events",
                    labels={
                        "datetime": "Datetime",
                        "ontario_avg_temp_c": "Avg Temp (°C)",
                        "temp_stress_type": "Stress Type",
                        "severity": "Severity"
                    }
                )
                st.plotly_chart(fig_stress, use_container_width=True)
            else:
                st.info("No stress events found in the selected range.")

        # Quick table
        preview_cols = [
            col for col in [
                "datetime", "ontario_avg_temp_c", "cold_stress",
                "heat_stress", "stress_event", "temp_stress_type", "severity"
            ] if col in filtered_df.columns
        ]
        if preview_cols:
            st.markdown("**Weather Context Preview**")
            st.dataframe(filtered_df[preview_cols].head(20), use_container_width=True)
            
#Tab 3: Early Warning Summary
with tab3:
    st.subheader("Monthly Summary and Top Events")

    if "year_month" in monthly_summary_df.columns and "stress_event_count" in monthly_summary_df.columns:
        fig_monthly = px.bar(
            monthly_summary_df,
            x="year_month",
            y="stress_event_count",
            title="Stress Events by Month",
            labels={"year_month": "Month", "stress_event_count": "Stress Event Count"}
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.warning("Monthly summary file does not contain expected columns.")

    # table
    if not top_events_df.empty:
        st.markdown("**Top Stress Events**")
        st.dataframe(top_events_df, use_container_width=True)
    else:
        st.info("Top stress events table is empty.")
        
#Download part
st.markdown("---")
st.subheader("Download Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Test Stress Events CSV",
    data=csv_data,
    file_name="filtered_test_stress_events.csv",
    mime="text/csv"
)

    
        