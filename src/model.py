import torch
import torch.nn as nn

class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, data: dict):
        """
        :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
                                                'logits', 'wt_logits', 'wt_embedding']
        :return: predicted DMS score
        """
        x = data['embedding']
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.sum(x, dim=1)
        x = self.fc2(x)
        return x