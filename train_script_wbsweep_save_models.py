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
import fcntl
import tempfile
from pathlib import Path
import json

#local imports
from model_arch import CategoricalScoreDiffusion
from otu_dataset import OTUDataset
from train_funcs import train_and_validate

def load_best_model(config):
    """Utility function to load the best model from W&B"""
    api = wandb.Api()
    artifact = api.artifact('project/best_model:latest')
    artifact_dir = artifact.download()
    model_path = Path(artifact_dir) / 'model.pt'
    
    # Initialize model with saved config
    model = CategoricalScoreDiffusion(**config)
    model.load_state_dict(torch.load(model_path))
    return model

# api = wandb.Api()
# artifact = api.artifact('matteopeluso1922/cdcd-hmp-param-search-local/best_model:latest')

def acquire_lock(lock_file):
    """Acquire an exclusive lock on the file"""
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return True
    except IOError:
        return False

def release_lock(lock_file):
    """Release the lock"""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def safe_read_best_loss(lock_file_path, loss_file_path):
    """Safely read the best loss with file locking"""
    with open(lock_file_path, 'w') as lock_file:
        acquire_lock(lock_file)
        try:
            if os.path.exists(loss_file_path):
                with open(loss_file_path, 'r') as f:
                    return float(f.read().strip())
            return float('inf')
        finally:
            release_lock(lock_file)

def safe_write_best_loss(lock_file_path, loss_file_path, loss):
    """Safely write the best loss with file locking"""
    with open(lock_file_path, 'w') as lock_file:
        acquire_lock(lock_file)
        try:
            with open(loss_file_path, 'w') as f:
                f.write(f"{loss}")
        finally:
            release_lock(lock_file)



def train_func():
    # Initialize wandb for this run
    run = wandb.init()
    #
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Calculate embed_dim as product to ensure validity
    embed_dim = config.head_dim * config.num_heads

    # Set up paths
    save_dir = Path("./model_checkpoints")
    save_dir.mkdir(exist_ok=True)
    loss_file_path = save_dir / "best_loss.txt"
    lock_file_path = save_dir / "best_loss.lock"
    
    
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
    
    # Read current best loss safely
    current_best_loss = safe_read_best_loss(lock_file_path, loss_file_path)
    
    # If we found a better model
    if best_val_loss < current_best_loss:
        # Safely write the new best loss
        safe_write_best_loss(lock_file_path, loss_file_path, best_val_loss)
        
        # Save model to a temporary file first
        temp_model_path = Path(tempfile.mktemp(suffix='.pt'))
        torch.save(model.state_dict(), temp_model_path)
        
        # Log model as W&B artifact
        artifact = wandb.Artifact(
            name=f'best_model_{run.id}', 
            type='model',
            metadata={
                'val_loss': best_val_loss,
                'config': dict(config)
            }
        )
        artifact.add_file(temp_model_path)
        run.log_artifact(artifact)
        
        # Clean up temporary file
        temp_model_path.unlink()
        
        # Save config
        config_path = save_dir / f"best_config_{best_val_loss:.6f}.json"
        with open(config_path, 'w') as f:
            json.dump(dict(config), f, indent=4)


    # Log the final metrics
    wandb.log({
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss
    })
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # Using Bayesian optimization
        "metric": {
            "name": "best_val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "head_dim": {  # dimension per attention head
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
                "max": 8
            },
         
            "dim_feedforward": {
                "distribution": "int_uniform",
                "min": 16,
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
                "value": 10
            },
            "vocab_size": {
                "value": None  # Will be set after loading data
            }
        }
    }

    # Load data once to get vocab_size
    loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')
    vocab_size = max(max(x) for x in loaded_df['otu_arrays']) + 1
    sweep_config["parameters"]["vocab_size"]["value"] = int(vocab_size)

    # Initialize sweep
    #sweep_id = wandb.sweep(sweep_config, project="cdcd-hmp-param-search-local")
    sweep_id = 'iyof9q52'
    # Start the sweep
    wandb.agent(sweep_id,project="cdcd-hmp-param-search-local",entity="matteopeluso1922", function=train_func, count=100)  # pass project and entity when using an existing sweep id.