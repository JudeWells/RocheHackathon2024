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


tok_to_idx = {'<cls>': 0,
              '<pad>': 1,
              '<eos>': 2,
              '<unk>': 3,
              'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8,
              'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13,
              'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18,
              'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23,
              'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28,
              '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

idx_to_tok = {v:k for k,v in tok_to_idx.items()}
class LikelihoodModel(nn.Module):
    """
    This model returns the logit (un-normalised likelihood) of the
    mutant residue as an estimator of the fitness score.
    It doesn't have any learnable parameters. In general,
    likelihood ratios MT/WT are probably better than doing this.
    """

    def __init__(self):
        super(LikelihoodModel, self).__init__()

    def get_mutated_position_idx(self, data):
        return [int(m[1:-1]) for m in data['mutant']]

    def get_mutant_aa_token_idx(self, data):
        return [tok_to_idx[m[-1]] for m in data['mutant']]

    def get_wt_aa_token_idx(self, data):
        return [tok_to_idx[m[0]] for m in data['mutant_sequence']]

    def forward(self, data: dict):
        print('Mutants:', data['mutant'])
        print('logits', data['logits'].shape)
        mutated_position_idx = self.get_mutated_position_idx(data)
        print('Positions:', mutated_position_idx)
        mutant_token_idx = self.get_mutant_aa_token_idx(data)
        wt_token_idx = self.get_wt_aa_token_idx(data)
        print('AA tokens:', mutant_token_idx)
        batch_indices = torch.arange(data['logits'].size(0))
        mutant_logit = data['wt_logits'][batch_indices, mutated_position_idx, mutant_token_idx]
        wt_logit = data['wt_logits'][batch_indices, mutated_position_idx, wt_token_idx]
        print('Mutant logits:', mutant_logit)
        return mutant_logit / wt_logit

