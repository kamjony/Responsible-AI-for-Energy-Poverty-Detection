import os
from src.config import ARTIFACTS_SAVE_PATH, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH, PLOTS_SAVE_PATH
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.prepare_sequences import prepare_lstm_inputs, save_artifacts
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_artifacts(input_dir: str = ARTIFACTS_SAVE_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    x_train_path = os.path.join(input_dir, "X_train.npy")
    x_val_path = os.path.join(input_dir, "X_val.npy")
    x_test_path = os.path.join(input_dir, "X_test.npy")
    scaler_path = os.path.join(input_dir, "demand_scaler.pkl")
    
    #Check if above files exist
    files_exist = all([
        os.path.exists(x_train_path),
        os.path.exists(x_val_path),
        os.path.exists(x_test_path),
        os.path.exists(scaler_path),
    ])
    
    if files_exist:
        print("\n:: LOADING EXISTING ARTIFACTS ::")
        X_train = np.load(x_train_path)
        X_val = np.load(x_val_path)
        X_test = np.load(x_test_path)
        scaler = joblib.load(scaler_path)
    else:
        print("\n:: ARTIFACTS NOT FOUND ::")
        print("Preparing LSTM inputs and saving artifacts...")

        X_train, X_val, X_test, scaler = prepare_lstm_inputs()
        save_artifacts(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            scaler=scaler,
            output_dir=input_dir
        )
    
    print("\n:: ARTIFACTS LOADED ::")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    
    return X_train, X_val, X_test, scaler

def build_lstm_autoencoder(sequence_length: int, n_features: int) -> Model:
    inputs = Input(shape=(sequence_length, n_features))
    
    #Encoder
    encoded = LSTM(BATCH_SIZE, activation="tanh", return_sequences=False)(inputs)
    bottleneck = Dense(16, activation="relu")(encoded)
    
    #Decoder
    repeated = RepeatVector(sequence_length)(bottleneck)
    decoded = LSTM(BATCH_SIZE, activation="tanh", return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    
    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    
    print("\n:: MODEL SUMMARY ::")
    model.summary()

    return model

def train_autoencoder(
    model: Model, 
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    epochs: int = EPOCHS, 
    batch_size: int = BATCH_SIZE, 
    output_dir: str = MODEL_SAVE_PATH):
    
    #Train LSTM autoencoder using train and validation sequences
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "lstm_autoencoder.keras")
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n:: TRAINING COMPLETE ::")
    print("Best validation loss:", min(history.history["val_loss"]))
    
    return history

def save_training_history(
    history, output_dir: str = MODEL_SAVE_PATH
) -> str:
    #Save training history as CSV
    
    os.makedirs(output_dir, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    
    print(f"Training history saved to: {history_path}")
    return history_path

def plot_training_loss(
    history,
    output_dir: str = PLOTS_SAVE_PATH
) -> str:
    #Train and validation loss curve plots
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_loss_curve.png")
    
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Training loss plot saved to: {plot_path}")
    return plot_path

def run_training(
    input_dir: str = ARTIFACTS_SAVE_PATH,
    model_output_dir: str = MODEL_SAVE_PATH,
    plot_output_dir: str = PLOTS_SAVE_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    #Load the artifacts
    X_train, X_val, X_test, scaler = load_artifacts(input_dir=input_dir)
    
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    #Build model
    model = build_lstm_autoencoder(sequence_length=sequence_length, n_features=n_features)
    
    #Train model
    history = train_autoencoder(
        model=model,
        X_train=X_train,
        X_val=X_val,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=model_output_dir
    )
    
    #Save history and plots
    save_training_history(history, output_dir=model_output_dir)
    plot_training_loss(history, output_dir=plot_output_dir)
    
    return model, history