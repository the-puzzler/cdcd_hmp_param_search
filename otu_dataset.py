import torch
from torch.utils.data import Dataset, DataLoader

class OTUDataset(Dataset):
   def __init__(self, df):
       self.df = df
       
       # Find max sequence length for padding
       self.max_len = max(len(x) for x in df['otu_arrays'])
       
   def __len__(self):
       return len(self.df)
   
   def __getitem__(self, idx):
       # Get array for this sample
       array = self.df.iloc[idx]['otu_arrays']
       
       # Create padded tensor
       padded = torch.zeros(self.max_len, dtype=torch.long)
       padded[:len(array)] = torch.tensor(array)
       
       # Create mask (False where we have real tokens, True for padding)
       mask = torch.zeros(self.max_len, dtype=torch.bool)
       mask[len(array):] = True
       
       return padded, mask