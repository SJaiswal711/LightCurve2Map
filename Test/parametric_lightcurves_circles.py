from EightBitTransit.TransitingImage import TransitingImage
import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import interp1d
import time

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
maps = np.load("FinalData/OM10/test_type0.npy")[0]
maps = maps[np.newaxis, :, :]
# plt.imshow(maps[0])
print(f"Loaded maps → shape={maps.shape}")


# --------------------------
# Parameter grids
# --------------------------
a_s = np.linspace(0.1, 0.9, 5)
b_s = np.linspace(0.05, 0.2, 4)
ratios = [0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05]

param_grid = list(itertools.product(a_s, b_s, ratios))
num_param_sets = len(param_grid)
print(f"Parameter combinations: {num_param_sets}")


from tqdm import tqdm

# --------------------------
# Wrapper to run all
# --------------------------
def generate_all_light_curves(maps, param_grid, n_times=1000, interp_len=150, pad_len=200):
    all_fluxes = []
    all_params = []
    all_indices = []

    num_maps = maps.shape[0]
    total_iters = num_maps * len(param_grid)

    with tqdm(total=total_iters, desc="Simulating LCs", ncols=100) as pbar:
        for m_idx in range(num_maps):
            for p_idx, (a, b, ratio) in enumerate(param_grid):
                _, flux = simulate_one_lc(
                    opacity_map=maps[m_idx],
                    v=0.4,
                    t_ref=0.0,
                    LDlaw='quadratic',
                    LDCs=[a, b],
                    star2mega_radius_ratio=1/ratio,  # <-- note reciprocal
                    n_times=n_times,
                    show=False
                )

                # --------------------------
                # 1. Interpolate to fixed length = interp_len
                # --------------------------
                x_old = np.linspace(0, 1, len(flux))
                x_new = np.linspace(0, 1, interp_len)
                f_interp = interp1d(x_old, flux, kind="linear")
                flux_interp = f_interp(x_new)

                # --------------------------
                # 2. Pad/crop to pad_len
                # --------------------------
                padded_flux = np.ones(pad_len, dtype=np.float32)
                L = min(len(flux_interp), pad_len)
                start = (pad_len - L) // 2
                padded_flux[start:start + L] = flux_interp[:L]

                all_fluxes.append(padded_flux)
                all_params.append([a, b, ratio])
                all_indices.append((m_idx, p_idx))

                pbar.update(1)

    return np.array(all_fluxes), np.array(all_params), np.array(all_indices)


# --------------------------
# Run simulation
# --------------------------
all_fluxes, all_params, all_indices = generate_all_light_curves(maps, param_grid)

print(f"Generated fluxes → shape={all_fluxes.shape}")
print(f"Params → shape={all_params.shape}")
print(f"Indices → shape={all_indices.shape}")

# --------------------------
# Save results
# --------------------------
np.savez("parametric_lightcurves_test_circles.npz",
         fluxes=all_fluxes,
         params=all_params,
         indices=all_indices)

print(" Saved → simulated_light_curves_circles.npz")
