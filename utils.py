import numpy as np

from scipy.interpolate import interp1d
def scale_vertically(ltcrvData):
    """
    Normalize each light curve (row) in the input data to the range [0, 1].

    Parameters
    ----------
    ltcrvData : np.ndarray
        A 2D array where each row represents a light curve.

    Returns
    -------
    np.ndarray
        The normalized 2D array where each row is scaled to [0, 1].
    """
    vltcrvData = np.zeros_like(ltcrvData)
    for i in range(len(ltcrvData)):
        min_val = np.min(ltcrvData[i])
        max_val = np.max(ltcrvData[i])
        vltcrvData[i] = (ltcrvData[i] - min_val) / (max_val - min_val)

    return vltcrvData


def scale_horizontally(ltcrvData, indices_array, target_length=120, kind='linear'):
    """
    Horizontally scale light curves using precomputed transit region indices.

    Parameters
    ----------
    ltcrvData : np.ndarray
        2D array of shape (N, S), light curves to process.
    indices_array : np.ndarray
        Array of shape (N, 3) with [left, right, region_length] for each light curve.
    target_length : int
        Desired length of the interpolated curves (default: 120).
    kind : str
        Type of interpolation ('linear', 'quadratic', etc.).

    Returns
    -------
    np.ndarray
        Interpolated light curves of shape (N, target_length). Skipped curves will be zero.
    """
    if ltcrvData.ndim != 2 or indices_array.ndim != 2 or indices_array.shape[1] != 3:
        raise ValueError("Invalid input shape. Expected (N, S) for ltcrvData and (N, 3) for indices_array.")

    N = ltcrvData.shape[0]
    interpolated_curves = np.zeros((N, target_length))

    for i in range(N):
        left, right, region_len = indices_array[i]

        if region_len < 3 or left < 0 or right > ltcrvData.shape[1]:
            print(f"[WARN] Skipping curve {i}: invalid region ({left}, {right},{region_len},{ltcrvData.shape[1]})")
            continue

        segment = ltcrvData[i, left:right]

        try:
            x_orig = np.linspace(-1, 1, num=len(segment))
            f = interp1d(x_orig, segment, kind=kind, fill_value="extrapolate")
            x_interp = np.linspace(-1, 1, num=target_length)
            interpolated_curves[i] = f(x_interp)
        except Exception as e:
            print(f"[ERROR] Interpolation failed for curve {i}: {str(e)}")
            continue

    return interpolated_curves

def Extend_ltcrv(ltcrvData,total_length = 150):
    """
    Extend each light curve in ltcrvData to length 150 by centering it and padding with 1s.
    
    Parameters
    ----------
    ltcrvData : np.ndarray
        Array of shape (N, T), where N is the number of light curves and T is their original length.
    
    Returns
    -------
    ltcrvData_append : np.ndarray
        Array of shape (N, 150), with each light curve centered and padded.
    """
    num_curves, orig_len = ltcrvData.shape
    # total_length = 500

    if orig_len > total_length:
        raise ValueError("Original light curves are longer than 150")

    # Initialize with ones
    ltcrvData_append = np.ones((num_curves, total_length))

    # Compute start index to center the original data
    start_idx = (total_length - orig_len) // 2
    end_idx = start_idx + orig_len

    # Fill each row centered
    ltcrvData_append[:, start_idx:end_idx] = ltcrvData

    return ltcrvData_append

def find_transit_regions(ltcrvs, threshold=0.99):
    """
    Given light curves:
    - Find in-transit points using the threshold
    - Compute region around transit as (center ± n/2 ± n/6)

    Parameters
    ----------
    scaled_ltcrvs : np.ndarray
        Array of shape (N, S), already scaled to [0, 1].
    threshold : float
        Flux threshold to detect in-transit points.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3): [left_idx, right_idx, region_length] per light curve.
        If insufficient transit points, returns [-1, -1, 0] for that curve.
    """
    N, S = ltcrvs.shape
    scaled_ltcrvs = scale_vertically(ltcrvs)
    results = np.zeros((N, 3), dtype=int)

    for i in range(N):
        curve = scaled_ltcrvs[i]

        # Find in-transit region
        mask = (curve < threshold).astype(np.float32)
        n = int(np.count_nonzero(mask == 1))
        if n < 3:
            results[i] = [-1, -1, 0]
            continue

        center = S // 2
        half_width = int(n*2 / 3)
        left = max(0, center - half_width)
        right = min(S, center + half_width)
        region_length = right - left

        results[i] = [left, right, region_length]

    return results

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

from scipy.ndimage import label, binary_closing, generate_binary_structure

def clean_prediction_mask(pred, threshold=0.5):
    """
    Post-process a batch of predictions by thresholding and extracting the largest connected component.
    
    Parameters
    ----------
    pred : np.ndarray of shape (N, 38, 38)
        Predicted probability masks (float32 values in [0, 1]).
    
    Returns
    -------
    np.ndarray of shape (N, 38, 38)
        Cleaned binary masks.
    """
    N = pred.shape[0]
    cleaned = np.zeros_like(pred)

    structure = generate_binary_structure(2, 2)  # 8-connected neighborhood

    for i in range(N):
        # Threshold to binary
        binary = pred[i] > threshold
        
        # Optional: morphological closing to fill small holes
        closed = binary_closing(binary, structure=structure)

        # Label connected components
        labeled, num = label(closed, structure=structure)

        if num == 0:
            continue  # skip if no region

        # Find largest connected component
        largest_cc = (labeled == np.argmax(np.bincount(labeled.flat)[1:]) + 1)
        cleaned[i] = largest_cc.astype(np.float32)

    return cleaned

def calculate_transit_depths_from_lc(lc_array):
    """
    Calculate transit depth from light curves.
    Depth = 1 - min(flux)
    lc_array: shape (N, T)
    Returns: shape (N,)
    """
    return 1.0 - np.min(lc_array, axis=1)
