from src.train_autoencoder import run_training
from src.config import ARTIFACTS_SAVE_PATH, MODEL_SAVE_PATH, PLOTS_SAVE_PATH, EPOCHS, BATCH_SIZE

def main():
    run_training(
        input_dir=ARTIFACTS_SAVE_PATH,
        model_output_dir=MODEL_SAVE_PATH,
        plot_output_dir=PLOTS_SAVE_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    

if __name__ == "__main__":
    main()