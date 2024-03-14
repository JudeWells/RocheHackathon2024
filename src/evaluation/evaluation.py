"""
This script should not be modified by participants
It is run as a github action to test if the participants code is working
and completes within the specified runtime.
"""

import os.path
import sys
import warnings
import time
# supress pandas deprication warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

from src.train import main as train_main

if __name__ == "__main__":
    start = time.time()
    rows = []
    # if command line argument provided, use that as the experiment name
    # otherwise, loop over all experiments
    if len(sys.argv) > 1:
        experiments = sys.argv[1].split(',')
    else:
        experiments = os.listdir('eval_files/')
    for experiment in experiments:
        try:
            experiment_path = 'eval_files/' + experiment.replace('.tar.gz', '')
            new_row = train_main(experiment_path=experiment_path, plot=False)
            rows.append(new_row)
        except Exception as e:
            print(f"Error with {experiment}: {e}")
            continue
    df = pd.DataFrame(rows)
    df.to_csv('supervised_results.csv', index=False)
    print(f"Metrics for {len(df)} experiments saved to supervised_results.csv")
    print(df.head())
    end = time.time()
    print(f"Total time: {(end-start)/60:.2f} minutes")