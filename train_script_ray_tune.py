
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

#ray tune
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Local imports
from loss_function import ce_loss_simple
import model_arch
from model_arch import CategoricalScoreDiffusion
from otu_dataset import OTUDataset
from train_funcs import train_and_validate



def train_func(config, train_dataset=None, test_dataset=None):
    # Initialize wandb for this trial
    run = wandb.init(
        project="cdcd-hmp-param-search",
        config={
            "learning_rate": config["lr"],
            "architecture": "",
            "dataset": "hmp",
            "epochs": config["num_epochs"],
            "embed_dim": config["embed_dim"],
            "num_layers": config["num_layer"],
            "num_heads": config["num_head"],
            "dim_feedforward": config["dim_ff"],
            "vocab_size": config["vocab_size"],
            "num_fourier_features": config["num_fourier"]
        },
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Initialize model with config
    model = CategoricalScoreDiffusion(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layer"],
        num_heads=config["num_head"],
        dim_feedforward=config["dim_ff"],
        num_fourier_features=config["num_fourier"]
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Train and validate
    best_val_loss, final_val_loss = train_and_validate(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        config["num_epochs"], 
        device,
        run.name  # Use wandb run name
    )
    
    # Clean up
    wandb.finish()
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Load data once
    loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')
    train_idx, test_idx = train_test_split(loaded_df.index, test_size=0.2, random_state=42)
    train_df = loaded_df.loc[train_idx]
    test_df = loaded_df.loc[test_idx]
    
    train_dataset = OTUDataset(train_df)
    test_dataset = OTUDataset(test_df)
    vocab_size = max(max(x) for x in loaded_df['otu_arrays']) + 1

    # Define search space
    search_space = {
        "embed_dim": tune.choice([4, 8, 16, 32]),
        "num_layer": tune.choice([1, 2, 3, 4, 5, 6]),
        "num_head": tune.choice([4, 8]),
        "dim_ff": tune.choice([16, 32, 64]),
        "num_fourier": tune.choice([2, 4, 8, 16, 32]),
        "num_epochs": 100,
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32]),
        "vocab_size": vocab_size
    }

    # Initialize Ray
    ray.init(num_cpus=32)  # Adjust based on your system

    # Setup scheduler for early stopping with proper metric and mode
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=10,
        reduction_factor=2,
        metric="val_loss",  
        mode="min"         
    )

    # Run optimization
    analysis = tune.run(
        tune.with_parameters(
            train_func,
            train_dataset=train_dataset,
            test_dataset=test_dataset
        ),
        config=search_space,
        num_samples=600,
        scheduler=scheduler,
        progress_reporter=tune.CLIReporter(
            parameter_columns=["embed_dim", "num_layer", "num_head", "lr", "batch_size"],
            metric_columns=["val_loss", "training_iteration"]
        ),
        name="transformer_tune"
        # Removed metric and mode as they're already in scheduler
    )

    # Get best config
    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    print("Best config:", best_config)