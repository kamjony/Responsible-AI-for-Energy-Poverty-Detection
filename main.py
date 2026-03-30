from src.config import *
from src.aggregate_results import run_aggregation

def main():
    run_aggregation(
        anomaly_results_dir=ANOMALY_SAVE_PATH,
        output_dir=AGGREGATED_RESULTS_SAVE_PATH
    )
    
    

if __name__ == "__main__":
    main()