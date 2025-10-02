import matplotlib.pyplot as plt

import numpy as np
import os
from utils import *

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


def process_and_save(lc_paths,snr, save_dir="FinalData/LC10/"):
    os.makedirs(save_dir, exist_ok=True)

    for lc_path in lc_paths:
        # Load light curves
        lcs = np.load(lc_path + ".npy")

        # Extend light curves
        elcs = Extend_ltcrv(lcs, total_length=125)

        # Identify transit regions
        indices = find_transit_regions(elcs, threshold=0.98)

        # Add noise with random SNR between 50 and 500
        # target_snr_array = np.random.uniform(100, 500, size=elcs.shape[0])
        target_snr_array = np.ones(elcs.shape[0]) * snr
        nlcs, _ = add_noise_to_bulk_lightcurves(elcs, target_snr_array, random_seed=42)
        
        # Scale vertically and horizontally
        vnlcs = scale_vertically(nlcs)
        FinalLc = scale_horizontally(vnlcs, indices)

        # Save to corresponding filename in save_dir
        base_name = os.path.basename(lc_path)
        save_path = os.path.join(save_dir, base_name + "LC.npy")
        np.save(save_path, FinalLc)

        print(f"✅ Saved processed LC to {save_path}")

TestLC1s = [
    "Data/LC/test_typeIa", "Data/LC/test_typeIb", "Data/LC/test_typeIIa",
    "Data/LC/test_typeIIb", "Data/LC/test_typeIII", "Data/LC/test_typeIV",
    "Data/LC/test_typeV"
]

CirLCs = ["Data/LC/test_type0"]

import os
SNRs = [50, 75, 100, 150, 200, 500]
for snr in SNRs:
    save_dir = f"FinalData/LC/{snr}/"
    os.makedirs(save_dir, exist_ok=True)
    process_and_save(TestLC1s,  snr=snr, save_dir=save_dir)
    process_and_save(CirLCs,    snr=snr, save_dir=save_dir)


