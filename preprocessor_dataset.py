import matplotlib.pyplot as plt

import numpy as np
import os
from utils import *

def calculate_transit_depths_from_lc(lc_array):
    """
    Calculate transit depth from light curves.
    Depth = 1 - min(flux)
    lc_array: shape (N, T)
    Returns: shape (N,)
    """
    return 1.0 - np.min(lc_array, axis=1)

def add_noise_to_bulk_lightcurves(lcs, target_snr_array, random_seed=None):
    """
    Add Gaussian noise to a batch of transit light curves, each with its own SNR.
    
    Parameters:
    -----------
    lcs : ndarray
        Array of light curves, shape (N, length).
    target_snr_array : ndarray
        Array of target SNR values, shape (N,).
    random_seed : int or None
        For reproducibility.
    
    Returns:
    --------
    noisy_lcs : ndarray
        Array of noisy light curves, same shape as input.
    sigma_noises : ndarray
        Array of noise standard deviations used for each light curve (N,).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N = lcs.shape[0]
    noisy_lcs = np.zeros_like(lcs)
    sigma_noises = np.zeros(N)

    for i in range(N):
        lc = lcs[i]
        snr = target_snr_array[i]
        
        depth = 1.0 - np.min(lc)
        if depth <= 0:
            raise ValueError(f"Transit depth must be positive in light curve {i}.")
        
        sigma_noise = depth / snr
        noise = np.random.normal(0, sigma_noise, size=lc.shape)
        
        noisy_lcs[i] = lc + noise
        sigma_noises[i] = sigma_noise
    
    return noisy_lcs, sigma_noises


def process_and_save(lc_paths, save_dir="FinalData/"):
    os.makedirs(save_dir, exist_ok=True)

    for lc_path in lc_paths:
        # Load light curves
        lcs = np.load(lc_path + ".npy")

        # Extend light curves
        elcs = Extend_ltcrv(lcs, total_length=125)

        # Identify transit regions
        indices = find_transit_regions(elcs, threshold=0.98)

        # Add noise with random SNR between 50 and 500
        target_snr_array = np.random.uniform(100, 500, size=elcs.shape[0])
        nlcs, _ = add_noise_to_bulk_lightcurves(elcs, target_snr_array, random_seed=42)
        
        # Scale vertically and horizontally
        vnlcs = scale_vertically(nlcs)
        FinalLc = scale_horizontally(vnlcs, indices)

        # Save to corresponding filename in save_dir
        base_name = os.path.basename(lc_path)
        save_path = os.path.join(save_dir, base_name + ".npy")
        np.save(save_path, FinalLc)

        print(f"✅ Saved processed LC to {save_path}")
        
ValLCs = [
    "Data/LC/val_typeIa", "Data/LC/val_typeIb", "Data/LC/val_typeIIa",
    "Data/LC/val_typeIIb", "Data/LC/val_typeIII", "Data/LC/val_typeIV",
    "Data/LC/val_typeV"
]
TrainLCs = [
    "Data/LC/train_typeIa", "Data/LC/train_typeIb", "Data/LC/train_typeIIa",
    "Data/LC/train_typeIIb", "Data/LC/train_typeIII", "Data/LC/train_typeIV",
    "Data/LC/train_typeV"
]

TestLCs = [
    "Data/LC/test_typeIa", "Data/LC/test_typeIb", "Data/LC/test_typeIIa",
    "Data/LC/test_typeIIb", "Data/LC/test_typeIII", "Data/LC/test_typeIV",
    "Data/LC/test_typeV"
]

CirLCs = ["Data/LC/train_type0","Data/LC/val_type0","Data/LC/test_type0"]

# Process train and val sets
process_and_save(TrainLCs, save_dir="FinalData/LC/")
process_and_save(ValLCs, save_dir="FinalData/LC/")
process_and_save(TestLCs, save_dir="FinalData/LC/")
process_and_save(CirLCs, save_dir="FinalData/LC/")

TrainOMs = [
    "train_typeIa", "train_typeIb", "train_typeIIa",
    "train_typeIIb", "train_typeIII", "train_typeIV",
    "train_typeV"
]
TestOMs = [
    "test_typeIa", "test_typeIb", "test_typeIIa",
    "test_typeIIb", "test_typeIII", "test_typeIV",
    "test_typeV"
]
ValOMs = [
    "val_typeIa", "val_typeIb", "val_typeIIa",
    "val_typeIIb", "val_typeIII", "val_typeIV",
    "val_typeV"
]

def repeat_and_save_shapes(shape_files, input_folder="Data/OM", output_folder="FinalData/OM"):
    os.makedirs(output_folder, exist_ok=True)

    for name in shape_files:
        print(f"\n📁 Processing shape file: {name}.npy")

        # Load opacity map
        path = os.path.join( input_folder, "Aug_"+name + ".npy")
        shape = np.load(path)

        # Repeat each shape 10 times along the first axis
        repeated_shape = np.repeat(shape, repeats=10, axis=0)

        # Save to output folder
        save_path = os.path.join(output_folder, name + ".npy")
        np.save(save_path, repeated_shape)

        print(f"✅ Saved to {save_path} | New shape: {repeated_shape.shape}")

# Lists of file names (without .npy extension)
TrainOMs = [
    "train_typeIa", "train_typeIb", "train_typeIIa",
    "train_typeIIb", "train_typeIII", "train_typeIV",
    "train_typeV"
]
TestOMs = [
    "test_typeIa", "test_typeIb", "test_typeIIa",
    "test_typeIIb", "test_typeIII", "test_typeIV",
    "test_typeV"
]
ValOMs = [
    "val_typeIa", "val_typeIb", "val_typeIIa",
    "val_typeIIb", "val_typeIII", "val_typeIV",
    "val_typeV"
]


# Process all datasets
repeat_and_save_shapes(TrainOMs)
repeat_and_save_shapes(TestOMs)
repeat_and_save_shapes(ValOMs)
