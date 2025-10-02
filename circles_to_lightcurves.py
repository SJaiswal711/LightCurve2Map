
from EightBitTransit.TransitingImage import TransitingImage

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
import copy
from functools import partial


import time

from scipy.interpolate import interp1d

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


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
                    star2mega_radius_ratio=10, n_times=10000, show=True):
    """
    Simulate a single light curve from an opacity map using EightBitTransit.
    """
    # Normalize and invert image
    opacity_map = opacity_map / np.max(opacity_map)

    # Pad opacity map to match star radius
    radius_mega = opacity_map.shape[0] / 2
    radius_star = radius_mega * star2mega_radius_ratio
    pad = int(radius_star - radius_mega)
    padded_map = np.pad(opacity_map, pad_width=((pad, pad), (6, 6)), mode='constant', constant_values=0.0)

    # Time array
    times = np.linspace(-35, 35, n_times)

    # Generate light curve
    if LDlaw is None:
        model = TransitingImage(opacitymat=padded_map,
                            v=v,
                            t_ref=t_ref,
                            t_arr=times)
    else:
        model = TransitingImage(opacitymat=padded_map,
                            v=v,
                            t_ref=t_ref,
                            t_arr=times,
                            LDlaw=LDlaw,
                            LDCs=LDCs)

    # ✅ Must pass t_arr explicitly
    flux, overlap_times = model.gen_LC(model.t_arr)
    # print(model)
    # Plot
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


def simulate_and_pad(index_tuple, maps, n_curves_per_map, fixed_ld_params):
    i, j = index_tuple
    np.random.seed(i * 1000 + j)

    opacity_map = maps[i]
    a, b = fixed_ld_params[i]  # Use fixed a, b for this map
    LDCs = [a, b]

    # Vary only the radius ratio per curve
    star2mega_radius_ratio = np.random.uniform(8.0, 20.0)

    _, flux = simulate_one_lc(
        opacity_map=opacity_map,
        v=0.4,
        t_ref=0.0,
        LDlaw='quadratic',
        LDCs=LDCs,
        star2mega_radius_ratio=star2mega_radius_ratio,
        n_times=1000,
        show=False
    )

    padded_flux = np.zeros(200, dtype=np.float32)
    L = len(flux)
    if L > 200:
        flux = flux[:200]
        L = 200
    start = (200 - L) // 2
    padded_flux[start:start + L] = flux

    return i * n_curves_per_map + j, padded_flux, [a, b, star2mega_radius_ratio]


def run_simulation(maps_path, save_path, n_curves_per_map=10):
    maps = np.load(maps_path)
    N = len(maps)

    # Generate fixed LD params per map (N values)
    fixed_ld_params = np.random.uniform(low=[0.1, 0.05], high=[0.9, 0.18], size=(N, 2))

    final_shape = (N * n_curves_per_map, 200)
    all_indices = [(i, j) for i in range(N) for j in range(n_curves_per_map)]
    all_lcs_padded = np.zeros(final_shape, dtype=np.float32)
    all_ld_params = np.zeros((N * n_curves_per_map, 3), dtype=np.float32)  # a, b, ratio

    start = time.time()
    with ProcessPoolExecutor() as executor:
        task = partial(simulate_and_pad, maps=maps, n_curves_per_map=n_curves_per_map, fixed_ld_params=fixed_ld_params)
        for idx, flux, ld_params in tqdm(executor.map(task, all_indices), total=len(all_indices)):
            all_lcs_padded[idx] = flux
            all_ld_params[idx] = ld_params
    end = time.time()

    print(f"Time taken for {maps_path}: {end - start:.4f} seconds")

    processed_lightcurves = process_all_lightcurves(all_lcs_padded, target_length=100)
    np.save(save_path, processed_lightcurves)
    print(f"Saved to {save_path}, Shape: {processed_lightcurves.shape}")

    meta_save_path = save_path.replace(".npy", "_meta.npy")
    np.save(meta_save_path, all_ld_params)
    print(f"Saved LDCs and radius ratios to {meta_save_path}, Shape: {all_ld_params.shape}")


run_simulation("Data/OM/type0.npy", "Data/LC/type0.npy")
