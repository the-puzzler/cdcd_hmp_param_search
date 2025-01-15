#####################################################
############## Importing ########################
#####################################################

# Standard library imports
import argparse
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

# Local imports
from loss_function import ce_loss_simple
import model_arch
from model_arch import CategoricalScoreDiffusion
from otu_dataset import OTUDataset
from train_funcs import train_and_validate

#####################################################
############## Param Initialisation #################
#####################################################

parser = argparse.ArgumentParser(description='Transformer model parameters')

parser.add_argument('--embed_dim', type=int, default=16,
                    help='Embedding dimension (default: 16)')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Number of transformer layers (default: 3)')
parser.add_argument('--num_heads', type=int, default=4,
                    help='Number of attention heads (default: 4)')
parser.add_argument('--dim_feedforward', type=int, default=16,
                    help='Dimension of feedforward network (default: 16)')
parser.add_argument('--num_fourier_features', type=int, required=True,
                    help='Number of Fourier features')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training (default: 32)')

args = parser.parse_args()

#####################################################
############## Data Loading ########################
#####################################################

# Load data
loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')

# Set random seed
np.random.seed(42)

# Split indices into train/test
train_idx, test_idx = train_test_split(loaded_df.index, test_size=0.2, random_state=42)

# Create train and test dataframes
train_df = loaded_df.loc[train_idx]
test_df = loaded_df.loc[test_idx]

# print(f"Train size: {len(train_df)}")
# print(f"Test size: {len(test_df)}")
# print("\nFirst few training samples:")
# print(train_df.head())

# Let's also look at array lengths
array_lengths = [len(x) for x in loaded_df['otu_arrays']]
# print(f"\nMin array length: {min(array_lengths)}")
# print(f"Max array length: {max(array_lengths)}")
# print(f"Mean array length: {np.mean(array_lengths):.2f}")


# Create datasets
train_dataset = OTUDataset(train_df)
test_dataset = OTUDataset(test_df)

# Create dataloaders
batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verify shapes
# for tokens, mask in train_loader:
#    print(f"Batch tokens shape: {tokens.shape}")
#    print(f"Batch mask shape: {mask.shape}")

#    break

# Get vocab size (maximum token ID + 1 for padding)
vocab_size = max(max(x) for x in loaded_df['otu_arrays']) + 1
# print(f"\nVocabulary size: {vocab_size}")


#####################################################
############## Model Initialisation #################
#####################################################

# Then use them as:
embed_dim = args.embed_dim
num_layers = args.num_layers
num_heads = args.num_heads
dim_feedforward = args.dim_feedforward
num_fourier_features = args.num_fourier_features
num_epochs = args.num_epochs
learning_rate = args.learning_rate


model = CategoricalScoreDiffusion(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
    num_fourier_features=num_fourier_features
    
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Move model to device
model = model.to(device)


#####################################################
############## WandB Initialisation #################
#####################################################



run = wandb.init(
    project="cdcd-hmp-param-search",
    config={
        "learning_rate": learning_rate,
        "architecture": "",
        "dataset": "hmp",
        "epochs": num_epochs,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dim_feedforward": dim_feedforward,
        "vocab_size": vocab_size,
        "num_fourier_features":num_fourier_features
    }
)

run_name = run.name

#####################################################
############## Train ################################
#####################################################

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Start training
train_and_validate(model, train_loader, test_loader, optimizer, num_epochs, device, run_name)