import os
import pandas as pd
import numpy as np
import joblib
from src.config import FEATURE_COL, MASTER_DATASET_PATH, SEQUENCE_LENGTH, ARTIFACTS_SAVE_PATH
from src.build_master_dataset import build_master_dataset, save_master_dataset
from sklearn.preprocessing import MinMaxScaler

def load_master_dataset(path: str = MASTER_DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        db = build_master_dataset()
        save_master_dataset(db)
        # raise FileNotFoundError(f"Master dataset not found in: {path}")
    #Load the merged master dataset
    dataset = pd.read_csv(path)
    
    if "datetime" not in dataset.columns:
        raise ValueError(f"Datetime column not found in master dataset")
    #Parse datetime correctly to datetime format
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])

     # Sort data & reset index
    dataset = dataset.sort_values("datetime").reset_index(drop=True)
    return dataset

def check_master_dataset(df: pd.DataFrame) -> None:
    #Basic checks before modeling
    print("\nMASTER DATASET CHECKS")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Datetime range:", df["datetime"].min(), "to", df["datetime"].max())
    print("Missing values:\n", df.isnull().sum())
    print("Data types: \n", df.dtypes)
    
    if "hourly_demand" not in df.columns:
        raise ValueError(f"hourly_demand column not found in master dataset")
    
    if df["hourly_demand"].isnull().any():
        raise ValueError(f"hourly_demand column has missing values in master dataset")
    #Count number of duplicate datetime
    duplicate_count = df["datetime"].duplicated().sum()
    print("Duplicate datetime count:", duplicate_count)
    
    if duplicate_count > 0:
        raise ValueError(f"Duplicate datetime values found in master dataset")

def split_dataset_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splitting dataset like below:
    **Train: 2018-01-01 to 2020-12-31
    **Validation: 2021-01-01 to 2021-12-31
    **Test: 2022-01-01 to 2022-12-31
    """
    
    train_df = df[(df["datetime"] >= "2018-01-01") & (df["datetime"] < "2021-01-01")].copy()
    val_df = df[(df["datetime"] >= "2021-01-01") & (df["datetime"] < "2022-01-01")].copy()
    test_df = df[(df["datetime"] >= "2022-01-01") & (df["datetime"] < "2023-01-01")].copy()
    
    print("\n::Master Dataset split::")
    print("Train shape:", train_df.shape, "| Range:", train_df["datetime"].min(), "to", train_df["datetime"].max())
    print("Val shape:  ", val_df.shape,   "| Range:", val_df["datetime"].min(), "to", val_df["datetime"].max())
    print("Test shape: ", test_df.shape,  "| Range:", test_df["datetime"].min(), "to", test_df["datetime"].max())
    
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more of the train/validation/test splits are empty.")
    
    return train_df, val_df, test_df

    
    
def scale_demand_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    feature_col: str = FEATURE_COL
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    As hourly_demand (feature) values are very large numbers like:
    13,000,000 or 18,000,000 (check master dataset in outputs folder).
    These raw large values can be a problem for NN. So scaling transforms
    them into a smaller consistent range like: 0.00, 0.45. This will 
    make models easier to train.
    """
    #Fit scaler during training and then transform validation and test
    """
    Scaling helps for stable training: NN learn by adjusting weights using gradients.
    If values are large, the gradients will be unstable, learning can become too slow
    or too erratic. So scaling makes the optimization process smoother.
    When inputs are all within a similar range, the model can learn faster.
    Scaling will help our autoencoder model's reconstruction error easier to work with
    """
    #MinMaxScaler rescales data to smallest value to 0 and largest value to 1
    
    scaler = MinMaxScaler()
    
    train_val = train_df[[feature_col]].values
    val_val = val_df[[feature_col]].values
    test_val = test_df[[feature_col]].values
    
    train_scaled = scaler.fit_transform(train_val)
    val_scaled = scaler.transform(val_val)
    test_scaled = scaler.transform(test_val)
    
    print("\n::Scaling summary::")
    print("Train scaled min/max:", train_scaled.min(), train_scaled.max())
    print("Val scaled min/max:  ", val_scaled.min(), val_scaled.max())
    print("Test scaled min/max: ", test_scaled.min(), test_scaled.max())
    
    return train_scaled, val_scaled, test_scaled, scaler
    
    
def create_sequences(data: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    data_length = len(data)
    if data_length < sequence_length:
        raise ValueError("Length of data is smaller than sequence length")
    
    #Rolling sequence for LSTM input
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
        
    #Output shape: (samples, timesteps, features)
    return np.array(sequences)
       

def prepare_lstm_inputs(
    master_dataset_path: str = MASTER_DATASET_PATH,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    #Load the master dataset
    df = load_master_dataset(master_dataset_path)
    
    #Do basic checks
    check_master_dataset(df)
    
    #Split dataset
    train_df, val_df, test_df = split_dataset_by_time(df)
    
    #Scale data
    train_scaled, val_scaled, test_scaled, scaler = scale_demand_data(
        train_df, val_df, test_df)
    
    #Create sequences for scaled data
    X_train = create_sequences(train_scaled, sequence_length=sequence_length)
    X_val = create_sequences(val_scaled, sequence_length=sequence_length)
    X_test = create_sequences(test_scaled, sequence_length=sequence_length)
    
    print("\n::SEQUENCE SUMMARY::")
    print("Sequence length:", sequence_length)
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    
    return X_train, X_val, X_test, scaler

def save_artifacts(
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    X_test: np.ndarray, 
    scaler: MinMaxScaler, 
    output_dir: str = ARTIFACTS_SAVE_PATH) -> None:
    #Save all numpy arrays and scaler for later use
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    joblib.dump(scaler, os.path.join(output_dir, "demand_scaler.pkl"))
    
    print(f"\nSaved artifacts to: {output_dir}")
    