import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

from utils.add_modulo_fold import assign_fold

class ProteinDataset(Dataset):
    def __init__(self, csv_file, root_dir, folds, transform=None, return_logits=False, return_wt=False):
        self.data_frame = pd.read_csv(csv_file)
        self.add_fold_to_df()
        self.data_frame = self.data_frame[self.data_frame['fold'].isin([str(f) for f in folds])]
        self.root_dir = root_dir
        self.transform = transform
        self.return_logits = return_logits
        self.return_wt = return_wt

    def add_fold_to_df(self):
        if 'fold' not in self.data_frame.columns:
            self.data_frame['fold'] = self.data_frame['mutant'].apply(assign_fold)
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mutant = self.data_frame.iloc[idx, 0]
        embedding_path = os.path.join(self.root_dir, 'embeddings', mutant + '.npy')
        embedding = np.load(embedding_path)
        embedding = torch.from_numpy(embedding).float()

        DMS_score = self.data_frame.iloc[idx, 2]
        DMS_score = torch.tensor(DMS_score).float()
        mutant_sequence = self.data_frame.iloc[idx, 1]

        sample = {'embedding': embedding, 'mutant': mutant, 'DMS_score': DMS_score, 'mutant_sequence': mutant_sequence}
        if self.return_logits:
            logits_path = os.path.join(self.root_dir, 'logits', mutant + '.npy')
            logits = np.load(logits_path)
            logits = torch.from_numpy(logits).float()
            sample['logits'] = logits
            if self.return_wt:
                wt_logits_path = os.path.join(self.root_dir, 'logits', 'wt.npy')
                wt_logits = np.load(wt_logits_path)
                wt_logits = torch.from_numpy(wt_logits).float()
                sample['wt_logits'] = wt_logits
        if self.return_wt:
            wt_embedding_path = os.path.join(self.root_dir, 'embeddings', 'wt.npy')
            wt_embedding = np.load(wt_embedding_path)
            wt_embedding = torch.from_numpy(wt_embedding).float()
            sample['wt_embedding'] = wt_embedding


        if self.transform:
            sample = self.transform(sample)

        return sample

def get_dataloader(root_dir, folds, batch_size=32, shuffle=True, return_logits=False, return_wt=False):
    csv_path = glob.glob(f"{root_dir}/*.csv")[0]
    dataset = ProteinDataset(csv_file=csv_path, root_dir=root_dir, folds=folds,
                             return_logits=return_logits, return_wt=return_wt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)