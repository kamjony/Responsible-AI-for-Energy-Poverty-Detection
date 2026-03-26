import pandas as pd

def preprocess_electricity_demand_data(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    # Convert date column
    df["datetime"] = pd.to_datetime(df["date"])
     # Ensure hour is integer
    df["hour"] = df["hour"].astype(int)
    
    # hour=1 means 00:00–01:00, so subtract 1 hour offset
    df["datetime"] = df["datetime"] + pd.to_timedelta(df["hour"] - 1, unit="h")
    #Drop columns
    # df = df.drop(columns=["date", "hour"])
    
    # Create date-only column for later merge
    df["date_only"] = df["datetime"].dt.date
    
    # # Sort data
    # df = df.sort_values("datetime").reset_index(drop=True)
    
    # Keep only 2018–2023 to match weather proxy
    df = df[(df["datetime"] >= "2018-01-01") & (df["datetime"] < "2024-01-01")]
    
    # columns to drop
    df = df.drop(columns=['date', 'hour'])
    
    # columns to move to front
    cols_to_front = ["datetime", "date_only"]
    
    # Get all other columns that are NOT in the front list
    other_cols = [c for c in df.columns if c not in cols_to_front]
    
    # Reorder the dataframe
    df = df[cols_to_front + other_cols]
    
    # Sort data & reset index
    df = df.sort_values("datetime").reset_index(drop=True)

    return df
    
def preprocess_weather_data(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Create date-only column
    df["date"] = df["date"].dt.date
    df = df.rename(columns={"date": "date_only"})
    
    # Keep only useful columns
    df = df[[
        "date_only",
        "ontario_avg_temp_c",
        "cold_stress",
        "heat_stress",
        "temp_stress_type"
    ]]

    # Sort data
    df = df.sort_values("date_only").reset_index(drop=True)

    
    # df = df.drop(columns=['toronto_mean_temp_c', 'ottawa_mean_temp_c', 
    #                                 'thunder_bay_mean_temp_c', 'sudbury_mean_temp_c', 
    #                                 'windsor_mean_temp_c', 'station_count'])

    return df
    