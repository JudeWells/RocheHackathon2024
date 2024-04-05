import os.path
import sys
import warnings
import time
# supress pandas deprication warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from data_loader import get_dataloader
from model import ProteinModel
DATA_DIR = 'data'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

def plot_metrics(metrics, title='Training Metrics'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, (k, v) in enumerate(metrics[0].items()):
        ax = axs[i // 2, i % 2]
        ax.plot([m[k] for m in metrics])
        ax.set_title(k)
    # add sup title
    fig.suptitle(title)
    save_path = f'{OUT_DIR}/{title}.png'
    plt.savefig(save_path)
    print(f"Saved plot of to {os.getcwd()}/{save_path}")
    plt.show()


def get_performance_metrics(predictions, actuals):
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    spearman = spearmanr(actuals, predictions)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'spearman_r': spearman.correlation}

def train_model(model, train_loader, validation_loader, plot=False):
    num_epochs = 10
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_epoch_metrics = []
    val_epoch_metrics = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pred_vals = []
        true_vals = []
        for i, data in enumerate(train_loader):
            labels = data['DMS_score']
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            pred_vals.extend(outputs.detach().numpy())
            true_vals.extend(labels.numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_metrics = get_performance_metrics(pred_vals, true_vals)
        train_epoch_metrics.append(train_metrics)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, MAE: {train_metrics["mae"]:.4f}, '
              f'R2: {train_metrics["r2"]:.4f}, Spearman R: {train_metrics["spearman_r"]:.4f}')
        val_metrics = evaluate_model(model, validation_loader)
        val_epoch_metrics.append(val_metrics)
    if plot:
        plot_metrics(train_epoch_metrics, title='Training Metrics')
        plot_metrics(val_epoch_metrics, title='Validation Metrics')


def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in test_loader:
            labels = data['DMS_score']
            outputs = model(data)
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    metrics = get_performance_metrics(predictions, actuals)
    return metrics

def main(experiment_path, train_folds=[1,2,3], validation_folds=[4], test_folds=[5], plot=True):
    print(f"\nTraining model on {experiment_path}")
    train_loader = get_dataloader(experiment_path=experiment_path, folds=train_folds, return_logits=True, return_wt=True)
    val_loader = get_dataloader(experiment_path=experiment_path, folds=validation_folds, return_logits=True)
    test_loader = get_dataloader(experiment_path=experiment_path, folds=test_folds, return_logits=True)
    model = ProteinModel()
    start = time.time()
    train_model(model, train_loader, val_loader, plot=plot)
    train_time = time.time() - start
    metrics = evaluate_model(model, test_loader)
    metrics['train_time_secs'] = round(train_time, 1)
    print("Test performance metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    metrics['DMS_id'] = experiment_path.split('/')[-1]
    return metrics

if __name__ == "__main__":
    start = time.time()
    rows = []
    # if command line argument provided, use that as the experiment name
    # otherwise, loop over all experiments
    if len(sys.argv) > 1:
        experiments = sys.argv[1].split(',')
    else:
        experiments = list(set([fname.split('.')[0] for fname in os.listdir(DATA_DIR) if not fname.startswith('.')]))
    for experiment in experiments:
        try:
            experiment_path = f"{DATA_DIR}/{experiment}"
            new_row = main(experiment_path=experiment_path)
            rows.append(new_row)
        except Exception as e:
            print(f"Error with {experiment}: {e}")
            continue
    df = pd.DataFrame(rows)
    df_savepath = f'{OUT_DIR}/supervised_results.csv'
    df.to_csv(df_savepath, index=False)
    print(f"Metrics for {len(df)} experiments saved to {os.getcwd()}/{df_savepath}")
    print(df.head())
    end = time.time()
    print(f"Total time: {(end-start)/60:.2f} minutes")