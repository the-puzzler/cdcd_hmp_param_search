# Standard library imports
import argparse
from collections import Counter
import os
os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"

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

# Define project constants
WANDB_PROJECT = "cdcd-hmp-param-search-remote"  # Your project name
WANDB_ENTITY = "matteopeluso1922"  # Your W&B username/entity


#local imports
from model_arch import CategoricalScoreDiffusion
from otu_dataset import OTUDataset
from train_funcs import train_and_validate


def train_func():
    # Initialize wandb for this run
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Calculate embed_dim as product to ensure validity
    embed_dim = config.head_dim * config.num_heads
    
    
    # Load data (moved inside function since each sweep run needs its own data)
    loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')
    train_idx, test_idx = train_test_split(loaded_df.index, test_size=0.2, random_state=42)
    train_df = loaded_df.loc[train_idx]
    test_df = loaded_df.loc[test_idx]
    
    train_dataset = OTUDataset(train_df)
    test_dataset = OTUDataset(test_df)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model with config
    model = CategoricalScoreDiffusion(
        vocab_size=config.vocab_size,
        embed_dim=embed_dim, #not from config, from multiplication above to ensure validity by construction
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dim_feedforward=config.dim_feedforward,
        num_fourier_features=config.num_fourier
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    best_val_loss, final_val_loss = train_and_validate(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        config.num_epochs, 
        device,
        run.name
    )
    
    # Log the final metrics
    wandb.log({
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss
    })
    
    del model
    torch.cuda.empty_cache()

def setup_new_sweep():
    """Create and initialize a new sweep, returning the sweep ID."""
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "best_val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "head_dim": {
                "distribution": "int_uniform",
                "min": 4,
                "max": 16
            },
            "num_heads": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 8
            },
            "num_layers": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 10
            },
            "dim_feedforward": {
                "distribution": "int_uniform",
                "min": 8,
                "max": 64
            },
            "num_fourier": {
                "distribution": "int_uniform",
                "min": 2,
                "max": 32
            },
            "learning_rate": {
                "distribution": "log_uniform",
                "min": np.log(1e-5),
                "max": np.log(1e-1)
            },
            "batch_size": {
                "distribution": "int_uniform",
                "min": 4,
                "max": 32
            },
            "num_epochs": {
                "value": 20
            },
            "vocab_size": {
                "value": None
            }
        }
    }

    # Load data once to get vocab_size
    loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')
    vocab_size = max(max(x) for x in loaded_df['otu_arrays']) + 1
    sweep_config["parameters"]["vocab_size"]["value"] = int(vocab_size)

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )
    print(f"SWEEP_ID_START:{sweep_id}:SWEEP_ID_END")
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run W&B sweep agent')
    parser.add_argument('--sweep_id', type=str, help='Existing sweep ID to continue', default=None)
    parser.add_argument('--count', type=int, help='Number of runs for this agent', default=500)
    parser.add_argument('--create_only', action='store_true', help='Only create sweep, don\'t run agent')
    args = parser.parse_args()

    # Initialize wandb
    wandb.login()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = setup_new_sweep()
        if args.create_only:
            exit(0)
    
    # Start the agent with explicit project and entity
    wandb.agent(
        sweep_id,
        function=train_func,
        count=args.count,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )