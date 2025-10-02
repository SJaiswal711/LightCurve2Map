from multiprocessing import Pool, cpu_count
from EightBitTransit.TransitingImage import TransitingImage
import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import interp1d
import time
from tqdm import tqdm

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

# --------------------------
# Load selected opacity maps
# --------------------------
maps = np.load("Data10/TestingSet/selected_maps.npy")
print(f"✅ Loaded maps → shape={maps.shape}")


# --------------------------
# Parameter grids
# --------------------------
a_s = np.linspace(0.1, 0.9, 5)
b_s = np.linspace(0.05, 0.2, 4)
ratios = [0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05]

param_grid = list(itertools.product(a_s, b_s, ratios))
num_param_sets = len(param_grid)
print(f"✅ Parameter combinations: {num_param_sets}")



# --------------------------
# Worker function for Pool
# --------------------------
def worker(args):
    m_idx, p_idx, maps, param_grid, n_times, interp_len, pad_len = args
    a, b, ratio = param_grid[p_idx]

    _, flux = simulate_one_lc(
        opacity_map=maps[m_idx],
        v=0.4,
        t_ref=0.0,
        LDlaw='quadratic',
        LDCs=[a, b],
        star2mega_radius_ratio=1/ratio,
        n_times=n_times,
        show=False
    )

    # 1. Interpolate
    x_old = np.linspace(0, 1, len(flux))
    x_new = np.linspace(0, 1, interp_len)
    f_interp = interp1d(x_old, flux, kind="linear")
    flux_interp = f_interp(x_new)

    # 2. Pad with 1’s
    padded_flux = np.ones(pad_len, dtype=np.float32)
    L = min(len(flux_interp), pad_len)
    start = (pad_len - L) // 2
    padded_flux[start:start + L] = flux_interp[:L]

    return padded_flux, [a, b, ratio], (m_idx, p_idx)


# --------------------------
# Parallel wrapper
# --------------------------
def generate_all_light_curves_parallel(maps, param_grid, n_times=1000, interp_len=150, pad_len=200):
    num_maps = maps.shape[0]
    total_iters = num_maps * len(param_grid)

    # prepare arguments
    tasks = [(m_idx, p_idx, maps, param_grid, n_times, interp_len, pad_len)
             for m_idx in range(num_maps)
             for p_idx in range(len(param_grid))]

    results = []
    with Pool(processes=cpu_count()) as pool:
        for r in tqdm(pool.imap_unordered(worker, tasks), total=total_iters, desc="Simulating LCs", ncols=100):
            results.append(r)

    # unzip results
    all_fluxes, all_params, all_indices = zip(*results)

    return np.array(all_fluxes), np.array(all_params), np.array(all_indices)


# --------------------------
# Run parallel simulation
# --------------------------
all_fluxes, all_params, all_indices = generate_all_light_curves_parallel(maps, param_grid)

print(f"✅ Generated fluxes → shape={all_fluxes.shape}")
print(f"✅ Params → shape={all_params.shape}")
print(f"✅ Indices → shape={all_indices.shape}")

# --------------------------
# Save results
# --------------------------
np.savez("parametric_lightcurves_test_shapes.npz",
         fluxes=all_fluxes,
         params=all_params,
         indices=all_indices)

print(" Saved → parametric_lightcurves_test_shapes.npz")
