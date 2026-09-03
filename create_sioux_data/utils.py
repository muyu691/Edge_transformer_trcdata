"""Utility functions for traffic-network dataset creation."""

import numpy as np
import torch


def compute_free_flow_times(G, speeds):
    """
    Compute free-flow travel times for each scenario.

    Formula: time (minutes) = length / speed * 60, where parsed network
    lengths and generated speeds are normalized to compatible distance/hour
    units by network_parser.py. For example, Anaheim TNTP Length (ft) and
    Speed (ft/min) are converted to miles and miles/hour at parse time.
    
    Args:
        G: NetworkX graph with normalized 'length' edge attribute
        speeds: [num_samples, num_edges] - speeds for each edge in each scenario
    
    Returns:
        free_flow_times: [num_samples, num_edges] - in minutes
    """
    num_samples = speeds.shape[0]
    edges = list(G.edges())
    num_edges = len(edges)
    
    free_flow_times = np.zeros((num_samples, num_edges))
    
    for i, (u, v) in enumerate(edges):
        length = G[u][v]['length']
        free_flow_times[:, i] = (length / speeds[:, i]) * 60
    
    return free_flow_times


def check_for_nans_and_infs(data_dict):
    """
    Check for NaN and Inf values in data arrays.
    
    Args:
        data_dict: Dictionary of {name: array}
    """
    print(f"\n{'='*60}")
    print("Checking for NaN and Inf values")
    print(f"{'='*60}")
    
    has_issues = False
    for name, data in data_dict.items():
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        num_nans = np.isnan(data).sum()
        num_infs = np.isinf(data).sum()
        
        if num_nans > 0 or num_infs > 0:
            print(f" {name}: {num_nans} NaNs, {num_infs} Infs")
            has_issues = True
        else:
            print(f"  {name}: No issues")
    
    if not has_issues:
        print(f"\nAll data clean!")
    else:
        print(f"\n Warning: Some data contains NaN or Inf values!")
    
    return not has_issues


def compute_statistics(data_dict):
    """
    Compute and print statistics for data arrays.
    
    Args:
        data_dict: Dictionary of {name: array}
    """
    print(f"\n{'='*60}")
    print("Data Statistics")
    print(f"{'='*60}")
    
    for name, data in data_dict.items():
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        print(f"\n  {name}:")
        print(f"    Shape: {data.shape}")
        print(f"    Min: {data.min():.4f}")
        print(f"    Max: {data.max():.4f}")
        print(f"    Mean: {data.mean():.4f}")
        print(f"    Std: {data.std():.4f}")
        print(f"    Median: {np.median(data):.4f}")


def create_data_directories():
    """
    Create necessary directories for storing processed data.
    """
    import os
    
    directories = [
        'processed_data',
        'processed_data/raw',
        'processed_data/processed',
        'processed_data/ood',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created/verified directory: {directory}")
