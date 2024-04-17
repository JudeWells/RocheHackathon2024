"""
This script should not be modified by participants
It is run as a github action to test if the participants code is working
and completes within the specified runtime.
"""
import random
import os.path
import sys
import warnings
import time
# supress pandas deprication warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

from src.train import train_model, evaluate_model
from src.data_loader import get_dataloader
from src.model import ProteinModel

def get_random_folds(experiment_name):
    """
    Assign the 5 folds based on the experiment name
    3 folds in train, 1 in validation, 1 in test
    :return:
    """
    """
    Assign the 5 folds based on the experiment name.
    3 folds in train, 1 in validation, 1 in test.

    :param experiment_name: str, name of the experiment to seed the random generator
    :return: tuple of lists, containing the train, validation, and test folds
    """
    folds = [1, 2, 3, 4, 5]
    # Seed the random number generator with the hash of the experiment name
    seed = sum([ord(c) for c in experiment_name])
    random.seed(seed)
    # Shuffle the list of folds
    random.shuffle(folds)
    # Assign 3 folds to train, 1 to validation, and 1 to test
    train = folds[:3]
    validation = [folds[3]]
    test = [folds[4]]
    return train, validation, test

if __name__ == "__main__":
    start = time.time()
    rows = []
    eval_directory = '../evaluation_set_embeddings'
    # if command line argument provided, use that as the experiment name
    # otherwise, loop over all experiments
    if len(sys.argv) > 1:
        experiments = sys.argv[1].split(',')
    else:
        experiments = os.listdir(eval_directory)
    for experiment in experiments:
        try:
            train_folds, validation_folds, test_folds = get_random_folds(experiment)
            experiment_path = os.path.join(eval_directory, experiment.replace('.tar.gz', ''))
            train_loader = get_dataloader(experiment_path=experiment_path, folds=train_folds, return_logits=True,
                                          return_wt=True)
            val_loader = get_dataloader(experiment_path=experiment_path, folds=validation_folds, return_logits=True)
            test_loader = get_dataloader(experiment_path=experiment_path, folds=test_folds, return_logits=True)
            model = ProteinModel()
            start = time.time()
            train_model(model, train_loader, val_loader, plot=False)
            test_metrics = evaluate_model(model, test_loader)
            train_eval_time = time.time() - start
            test_metrics['DMS_id'] = experiment
            test_metrics['train_and_eval_time_secs'] = round(train_eval_time, 1)
            rows.append(test_metrics)
        except Exception as e:
            print(f"Error with {experiment}: {e}")
            continue
    df = pd.DataFrame(rows)
    df.to_csv('supervised_results.csv', index=False)
    print(f"Metrics for {len(df)} experiments saved to supervised_results.csv")
    print(df.head())
    end = time.time()
    print(f"Total time: {(end-start)/60:.2f} minutes")