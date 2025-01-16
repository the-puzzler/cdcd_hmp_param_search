import torch
from torch.utils.data import Dataset, DataLoader

class OTUDataset(Dataset):
    def __init__(self, df):
        self.max_len = max(len(x) for x in df['otu_arrays'])
        
        # Precompute all padded tensors and masks
        n_samples = len(df)
        self.padded_arrays = torch.zeros((n_samples, self.max_len), dtype=torch.long)
        self.masks = torch.zeros((n_samples, self.max_len), dtype=torch.bool)
        
        # Fill tensors
        for i, array in enumerate(df['otu_arrays']):
            self.padded_arrays[i, :len(array)] = torch.tensor(array)
            self.masks[i, len(array):] = True
            
    def __len__(self):
        return len(self.padded_arrays)
    
    def __getitem__(self, idx):
        return self.padded_arrays[idx], self.masks[idx]