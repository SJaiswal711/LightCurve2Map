
from EightBitTransit.TransitingImage import TransitingImage

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import interp1d
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import random

def get_random_ldc():
    """
    Returns a randomly selected (a, b) quadratic limb darkening coefficient pair
    from a representative sample (Teff = 5000–6500 K, log(g) = 4.5).

    Returns
    -------
    a, b : float
        Quadratic limb darkening coefficients.
    """
    ldc_values = [
        (0.38, 0.27),  # Teff = 5000 K
        (0.36, 0.28),  # Teff = 5500 K
        (0.34, 0.30),  # Teff = 6000 K
        (0.32, 0.31),  # Teff = 6500 K
    ]

    return random.choice(ldc_values)


def extract_and_interpolate(lc, target_length=100):
    non_zero = np.nonzero(lc)[0]
    if len(non_zero) == 0:
        return np.zeros(target_length)

    start, end = non_zero[0], non_zero[-1] + 1
    transit = lc[start:end]

    x_orig = np.linspace(0, 1, len(transit))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_orig, transit, kind='linear')
    return interpolator(x_new)

def process_all_lightcurves(lc_array, target_length=100):
    N = lc_array.shape[0]
    processed = np.ones((N, target_length), dtype=np.float32)
    for i in range(N):
        processed[i] = extract_and_interpolate(lc_array[i], target_length)
    return processed

def simulate_one_lc(opacity_map, v=0.4, t_ref=0.0, LDlaw='quadratic', LDCs=[0.3, 0.2],
                    star2mega_radius_ratio=10, n_times=20000, show=False):
    opacity_map = opacity_map / np.max(opacity_map)
    radius_mega = opacity_map.shape[0] / 2
    radius_star = radius_mega * star2mega_radius_ratio
    pad = int(radius_star - radius_mega)
    padded_map = np.pad(opacity_map, pad_width=((pad, pad), (6, 6)), mode='constant', constant_values=0.0)
    times = np.linspace(-35, 35, n_times)

    model = TransitingImage(
        opacitymat=padded_map,
        v=v,
        t_ref=t_ref,
        t_arr=times,
        LDlaw=LDlaw,
        LDCs=LDCs
    )
    flux, overlap_times = model.gen_LC(model.t_arr)

    if show:
        plt.figure(figsize=(8, 4))
        plt.plot(overlap_times, flux, label="Simulated LC")
        plt.xlabel("Time [days]")
        plt.ylabel("Normalized Flux")
        plt.ylim(min(flux) - 0.01, 1.01)
        plt.title("Transit Light Curve with Quadratic LD")
        plt.grid(True)
        plt.legend()
        plt.show()

    return overlap_times, flux

def simulate_and_pad_fixed_params(index_tuple, maps, counts, param_sets, num_simulations):
    i, j = index_tuple
    count_before = np.sum(counts[:i])
    param = param_sets[i][j]
    a, b, ratio = param
    LDCs = [a, b]

    outputs = []
    for k in range(counts[i]):
        _, flux = simulate_one_lc(
            opacity_map=maps[count_before + k],
            v=0.4,
            t_ref=0.0,
            LDlaw='quadratic',
            LDCs=LDCs,
            star2mega_radius_ratio=ratio,
            n_times=1000,
            show=False
        )

        padded_flux = np.zeros(200, dtype=np.float32)
        L = min(len(flux), 200)
        start = (200 - L) // 2
        padded_flux[start:start + L] = flux[:L]

        # Clean index: shape-major, sim-major, aug-minor
        # final_idx = (i * num_simulations * counts[i]) + (j * counts[i]) + k
        final_idx = (np.sum(counts[:i]) + k) * num_simulations + j

        outputs.append((final_idx, padded_flux, [a, b, ratio]))

    return outputs

def run_simulation_fixed_params(maps_path, counts_path, save_path, num_simulations=25):
    maps = np.load(maps_path)              # (M, H, W)
    counts = np.load(counts_path)          # (N,)
    total_shapes = len(counts)
    total_lc = np.sum(counts) * num_simulations

    param_sets = []
    for _ in range(total_shapes):
        shape_params = []
        for _ in range(num_simulations):
            a = np.random.uniform(0.1, 0.9)
            b = np.random.uniform(0.05, 0.2)
            # a,b = get_random_ldc()
            ratio = np.random.uniform(5.0, 20.0)
            shape_params.append([a, b, ratio])
        param_sets.append(shape_params)

    task_indices = [(i, j) for i in range(total_shapes) for j in range(num_simulations)]
    all_lcs_padded = np.zeros((total_lc, 200), dtype=np.float32)
    all_ld_params = np.zeros((total_lc, 3), dtype=np.float32)

    start = time.time()
    with ProcessPoolExecutor() as executor:
        task = partial(simulate_and_pad_fixed_params, maps=maps, counts=counts, param_sets=param_sets, num_simulations=num_simulations)
        for result_list in tqdm(executor.map(task, task_indices), total=len(task_indices)):
            for idx, flux, ld_params in result_list:
                all_lcs_padded[idx] = flux
                all_ld_params[idx] = ld_params
    end = time.time()
    print(f"⏱️ Time taken for {maps_path}: {end - start:.2f} seconds")

    processed_lightcurves = process_all_lightcurves(all_lcs_padded, target_length=100)
    print(f"✅ Processed LC shape: {processed_lightcurves.shape}")

    if not save_path.endswith(".npy"):
        save_path = save_path + ".npy"

    np.save(save_path, processed_lightcurves)
    print(f"✅ Saved LC to {save_path}")

    meta_path = save_path.replace(".npy", "_meta.npy")
    np.save(meta_path, all_ld_params)
    print(f"✅ Saved LD meta to {meta_path}")

run_simulation_fixed_params("Data/OM/Aug_test_typeIa.npy", "Data/OM/Num_test_typeIa.npy", "Data/LC0/test_typeIa", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_test_typeIb.npy", "Data/OM/Num_test_typeIb.npy", "Data/LC0/test_typeIb", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_test_typeIIa.npy", "Data/OM/Num_test_typeIIa.npy", "Data/LC0/test_typeIIa", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_test_typeIIb.npy", "Data/OM/Num_test_typeIIb.npy", "Data/LC0/test_typeIIb", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_test_typeIII.npy", "Data/OM/Num_test_typeIII.npy", "Data/LC0/test_typeIII", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_test_typeIV.npy", "Data/OM/Num_test_typeIV.npy", "Data/LC0/test_typeIV", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_test_typeV.npy", "Data/OM/Num_test_typeV.npy", "Data/LC0/test_typeV", num_simulations=10)


run_simulation_fixed_params("Data/OM/Aug_val_typeIa.npy", "Data/OM/Num_val_typeIa.npy", "Data/LC0/val_typeIa", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_val_typeIb.npy", "Data/OM/Num_val_typeIb.npy", "Data/LC0/val_typeIb", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_val_typeIIa.npy", "Data/OM/Num_val_typeIIa.npy", "Data/LC0/val_typeIIa", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_val_typeIIb.npy", "Data/OM/Num_val_typeIIb.npy", "Data/LC0/val_typeIIb", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_val_typeIII.npy", "Data/OM/Num_val_typeIII.npy", "Data/LC0/val_typeIII", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_val_typeIV.npy", "Data/OM/Num_val_typeIV.npy", "Data/LC0/val_typeIV", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_val_typeV.npy", "Data/OM/Num_val_typeV.npy", "Data/LC0/val_typeV", num_simulations=10)

run_simulation_fixed_params("Data/OM/Aug_train_typeIa.npy", "Data/OM/Num_train_typeIa.npy", "Data/LC/train_typeIa", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_train_typeIb.npy", "Data/OM/Num_train_typeIb.npy", "Data/LC/train_typeIb", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_train_typeIIa.npy", "Data/OM/Num_train_typeIIa.npy", "Data/LC/train_typeIIa", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_train_typeIIb.npy", "Data/OM/Num_train_typeIIb.npy", "Data/LC/train_typeIIb", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_train_typeIII.npy", "Data/OM/Num_train_typeIII.npy", "Data/LC/train_typeIII", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_train_typeIV.npy", "Data/OM/Num_train_typeIV.npy", "Data/LC/train_typeIV", num_simulations=10)
run_simulation_fixed_params("Data/OM/Aug_train_typeV.npy", "Data/OM/Num_train_typeV.npy", "Data/LC/train_typeV", num_simulations=10)
