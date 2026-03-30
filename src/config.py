FEATURE_COL = "hourly_demand"
SEQUENCE_LENGTH = 24
MASTER_DATASET_PATH = "outputs/master_dataset_2018_2023.csv"
ARTIFACTS_SAVE_PATH = "outputs/processed_artifacts"

#Model
BATCH_SIZE: int = 32
EPOCHS: int = 50
MODEL_SAVE_PATH = "outputs/models"
LSTM_SAVED_MODEL_PATH = "outputs/models/lstm_autoencoder.keras"

#Plots
PLOTS_SAVE_PATH = "outputs/plots"

#Anomaly
ANOMALY_SAVE_PATH = "outputs/anomaly_results"
THRESHOLD_PERCENTILE: float = 95.0

#Aggregated results
AGGREGATED_RESULTS_SAVE_PATH = "outputs/aggregated_results"
