import os
import pandas as pd
import numpy as np
from src.config import *
from src.detect_anomalies import run_detection

def load_anomaly_results(
    input_dir: str = ANOMALY_SAVE_PATH,
    artifacts_dir: str = ARTIFACTS_SAVE_PATH,
    model_path: str = LSTM_SAVED_MODEL_PATH,
    threshold_percentile: float = THRESHOLD_PERCENTILE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #Load anomaly result CSVs.
    train_path = os.path.join(input_dir, "train_anomaly_results.csv")
    val_path = os.path.join(input_dir, "val_anomaly_results.csv")
    test_path = os.path.join(input_dir, "test_anomaly_results.csv")
    
    files_exist = all([
        os.path.exists(train_path),
        os.path.exists(val_path),
        os.path.exists(test_path),
    ])
    
    if not files_exist:
        print("\n:: ANOMALY RESULT FILES NOT FOUND ::")
        print("Running detection to generate anomaly result files...")
        
        run_detection(
            artifacts_dir=artifacts_dir,
            model_path=model_path,
            threshold_percentile=threshold_percentile
        )
        
    print("\n:: LOADING ANOMALY RESULT FILES ::")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, val_df, test_df]:
        df["datetime"] = pd.to_datetime(df["datetime"])
        
    print("\n:: ANOMALY RESULTS LOADED ::")
    print("Train shape:", train_df.shape)
    print("Val shape:  ", val_df.shape)
    print("Test shape: ", test_df.shape)
    
    return train_df, val_df, test_df

def create_stress_event_flag(df: pd.DataFrame) -> pd.DataFrame:
    #Creates a binary stress flag
    #stress event = anomaly flag == 1 AND (cold stress == 1 OR heat stress == 1)
    result_df = df.copy()
    
    result_df["stress_event"] = (
        (result_df["anomaly_flag"] == 1) &
        ((result_df["cold_stress"] == 1) | (result_df["heat_stress"] == 1))
    ).astype(int)
    
    return result_df

def assign_severity_levels(df: pd.DataFrame) -> pd.DataFrame:
    #Assign severity levels based on reconstruction error among anomaly rows
    #Only anomality rows get a severity; non-anomalies get "None"
    
    result_df = df.copy()
    result_df["severity"] = "None"
    
    anomaly_mask = result_df["anomaly_flag"] == 1
    
    if anomaly_mask.sum() == 0:
        return result_df
    
    anomaly_errors = result_df.loc[anomaly_mask, "reconstruction_error"]
    
    low_cut = anomaly_errors.quantile(0.33)
    high_cut = anomaly_errors.quantile(0.66)
    
    result_df.loc[anomaly_mask & (result_df["reconstruction_error"] <= low_cut), "severity"] = "Low"
    result_df.loc[
        anomaly_mask &
        (result_df["reconstruction_error"] > low_cut) &
        (result_df["reconstruction_error"] <= high_cut),
        "severity"
    ] = "Medium"
    result_df.loc[anomaly_mask & (result_df["reconstruction_error"] > high_cut), "severity"] = "High"

    return result_df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    #Add year/month/date features for aggregation
    result_df = df.copy()
    result_df["year"] = result_df["datetime"].dt.year
    result_df["month"] = result_df["datetime"].dt.month
    result_df["year_month"] = result_df["datetime"].dt.to_period("M").astype(str)
    result_df["date"] = result_df["datetime"].dt.date

    return result_df

def create_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    #Create a monthly summary for all the anomalies and stress events
    summary_df = df.copy()
    
    monthly_summary = summary_df.groupby("year_month").agg(
        total_rows=("datetime", "count"),
        anomaly_count=("anomaly_flag", "sum"),
        stress_event_count=("stress_event", "sum"),
        cold_stress_rows=("cold_stress", "sum"),
        heat_stress_rows=("heat_stress", "sum"),
        avg_reconstruction_error=("reconstruction_error", "mean"),
        max_reconstruction_error=("reconstruction_error", "max")
    ).reset_index()

    monthly_summary["anomaly_rate_pct"] = (
        monthly_summary["anomaly_count"] / monthly_summary["total_rows"] * 100
    )
    monthly_summary["stress_event_rate_pct"] = (
        monthly_summary["stress_event_count"] / monthly_summary["total_rows"] * 100
    )

    return monthly_summary

def extract_stress_events(df: pd.DataFrame) -> pd.DataFrame:
    #Get rows only that are marked as stress events
    event_df = df[df["stress_event"] == 1].copy()
    event_df = event_df.sort_values("reconstruction_error", ascending=False).reset_index(drop=True)
    return event_df

def create_top_events_table(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    #Create table with stress events sorted by reconstruction error
    stress_events_df = extract_stress_events(df)
    
    if stress_events_df.empty:
        return stress_events_df
    
    top_events = stress_events_df[[
        "datetime",
        "hourly_demand",
        "hourly_average_price",
        "ontario_avg_temp_c",
        "cold_stress",
        "heat_stress",
        "temp_stress_type",
        "reconstruction_error",
        "severity"
    ]].head(top_n).copy()

    return top_events

def print_summary(name: str, df: pd.DataFrame):
    #Print summary of statistics
    anomaly_count = int(df["anomaly_flag"].sum())
    stress_event_count = int(df["stress_event"].sum())
    total_rows = len(df)

    print(f"\n:: {name.upper()} Statistics SUMMARY ::")
    print("Total rows:", total_rows)
    print("Anomalies:", anomaly_count)
    print("Stress events:", stress_event_count)

    if anomaly_count > 0:
        print("Stress events among anomalies (%):", round((stress_event_count / anomaly_count) * 100, 2))
    else:
        print("Stress events among anomalies (%): 0.0")

    if stress_event_count > 0:
        print("Stress type breakdown:")
        print(df[df["stress_event"] == 1]["temp_stress_type"].value_counts())
        print("Severity breakdown:")
        print(df[df["stress_event"] == 1]["severity"].value_counts())
        
def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_monthly: pd.DataFrame,
    val_monthly: pd.DataFrame,
    test_monthly: pd.DataFrame,
    test_top_events: pd.DataFrame,
    output_dir: str = AGGREGATED_RESULTS_SAVE_PATH
):
    #Save all outputs
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train_stress_events.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_stress_events.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_stress_events.csv"), index=False)

    train_monthly.to_csv(os.path.join(output_dir, "train_monthly_summary.csv"), index=False)
    val_monthly.to_csv(os.path.join(output_dir, "val_monthly_summary.csv"), index=False)
    test_monthly.to_csv(os.path.join(output_dir, "test_monthly_summary.csv"), index=False)

    test_top_events.to_csv(os.path.join(output_dir, "test_top_stress_events.csv"), index=False)

    print(f"\nSaved Phase 5 outputs to: {output_dir}")
    
def run_aggregation(
    anomaly_results_dir: str = ANOMALY_SAVE_PATH,
    output_dir: str = AGGREGATED_RESULTS_SAVE_PATH,
):
    #Load anomaly results
    train_df, val_df, test_df = load_anomaly_results(anomaly_results_dir)
    
    #Create stress events
    train_df = create_stress_event_flag(train_df)
    val_df = create_stress_event_flag(val_df)
    test_df = create_stress_event_flag(test_df)
    
    #Assign severity
    train_df = assign_severity_levels(train_df)
    val_df = assign_severity_levels(val_df)
    test_df = assign_severity_levels(test_df)
    
    #Add time features
    train_df = add_time_features(train_df)
    val_df = add_time_features(val_df)
    test_df = add_time_features(test_df)
    
    #Monthly summaries
    train_monthly = create_monthly_summary(train_df)
    val_monthly = create_monthly_summary(val_df)
    test_monthly = create_monthly_summary(test_df)
    
    #Top event table 
    test_top_events = create_top_events_table(test_df, top_n=25)
    
    #Save
    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_monthly=train_monthly,
        val_monthly=val_monthly,
        test_monthly=test_monthly,
        test_top_events=test_top_events,
        output_dir=output_dir
    )
    
    return train_df, val_df, test_df, train_monthly, val_monthly, test_monthly, test_top_events


        