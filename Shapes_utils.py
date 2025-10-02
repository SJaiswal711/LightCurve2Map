import numpy as np
from bezier_utils import *
import numpy as np
from scipy.spatial.distance import hamming
from skimage.draw import polygon

"""-------- Generate the Shapes --------"""

def generate_circles(num_maps=1000, size=38):
    radius = size // 2
    center = (size - 1) / 2

    y, x = np.ogrid[:size, :size]
    dist_from_center = (x - center)**2 + (y - center)**2
    mask = dist_from_center <= radius**2

    single_circle = np.zeros((size, size), dtype=np.uint8)
    single_circle[mask] = 1

    return np.repeat(single_circle[np.newaxis, :, :], num_maps, axis=0)


def generate_perturbed_circles(n_points=5000, n_modes=5, radial_base=1.0, scale=0.3, seed=None):
    """
    Generate a smooth, closed shape using random Fourier perturbations.
    
    Parameters:
    - n_points: Number of points along the shape
    - n_modes: Number of Fourier modes (higher = more complex)
    - radial_base: Base radius
    - scale: Magnitude of perturbations
    - seed: Random seed for reproducibility
    
    Returns:
    - x, y: Arrays of shape coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = np.ones_like(theta) * radial_base

    # Add random Fourier modes
    for k in range(1, n_modes + 1):
        amplitude = np.random.uniform(-scale, scale)
        phase = np.random.uniform(0, 2 * np.pi)
        r += amplitude * np.cos(k * theta + phase)

    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Normalize to [0, 1] for consistency
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    return x, y


def generate_perturbed_ellipses(n_points=500, a=0.6, b=1.0, n_modes=6, scale=0.1, orientation=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Base ellipse
    x_base = a * np.cos(theta)
    y_base = b * np.sin(theta)

    # Radial perturbation
    perturbation = np.zeros_like(theta)
    for k in range(1, n_modes + 1):
        amplitude = np.random.uniform(-scale, scale)
        phase = np.random.uniform(0, 2 * np.pi)
        perturbation += amplitude * np.sin(k * theta + phase)

    r_perturbed = 1 + perturbation
    x = r_perturbed * x_base
    y = r_perturbed * y_base

    # Apply orientation rotation
    cos_angle = np.cos(orientation)
    sin_angle = np.sin(orientation)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y

    # Scale uniformly to fit within [0,1] while preserving aspect ratio
    x_range = np.max(x_rot) - np.min(x_rot)
    y_range = np.max(y_rot) - np.min(y_rot)
    scale_factor = 0.9 / max(x_range, y_range)  # 90% fill for margin
    x_scaled = x_rot * scale_factor
    y_scaled = y_rot * scale_factor

    # Center inside [0,1] box
    x_centered = x_scaled - np.min(x_scaled)
    y_centered = y_scaled - np.min(y_scaled)

    x_centered = x_centered + (1.0 - np.max(x_centered)) / 2
    y_centered = y_centered + (1.0 - np.max(y_centered)) / 2

    return x_centered, y_centered


def generate_symmetric_shapes(n_points=2, scale=0.8, rad=0.2, edgy=0.0, mirror='vertical', seed=None):
    if seed is not None:
        np.random.seed(seed)

    half_points = get_random_points(n=n_points, scale=scale)
    
    if mirror == 'vertical':
        half_points[:, 0] = -np.abs(half_points[:, 0])
        mirrored_points = half_points.copy()
        mirrored_points[:, 0] *= -1
    elif mirror == 'horizontal':
        half_points[:, 1] = -np.abs(half_points[:, 1])
        mirrored_points = half_points.copy()
        mirrored_points[:, 1] *= -1
    else:
        raise ValueError("mirror must be 'vertical' or 'horizontal'")

    full_points = np.concatenate([half_points, mirrored_points[::-1]])
    x, y, _ = get_bezier_curve(full_points, rad=rad, edgy=edgy)
    x = x / 2

    # Normalize to [0, 1]^2, preserving aspect ratio
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    scale_factor = 1.0 / max(x_range, y_range)
    x = (x - np.min(x)) * scale_factor
    y = (y - np.min(y)) * scale_factor
    x_pad = (1 - (np.max(x) - np.min(x))) / 2
    y_pad = (1 - (np.max(y) - np.min(y))) / 2
    x += x_pad - np.min(x)
    y += y_pad - np.min(y)

    return x, y


def shape_to_mask(x, y, size=38):
    rr, cc = polygon(y * (size - 1), x * (size - 1), shape=(size, size))
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask

"""-------- Unique Shapes --------"""


def get_unique_shapes1(grids, threshold=0.02):
    """
    Identify and return unique 38x38 binary shape grids using a similarity threshold.

    Parameters
    ----------
    grids : np.ndarray
        Array of shape (N, 38, 38), where each grid is a binary mask (0/1).
    threshold : float
        Hamming distance threshold (0 to 1). Lower means stricter uniqueness.

    Returns
    -------
    unique_count : int
        Number of unique shapes.
    unique_grids : np.ndarray
        Array of unique shape grids with shape (M, 38, 38), M <= N.
    """
    grids = np.asarray(grids).astype(np.uint8)
    flat_grids = grids.reshape((grids.shape[0], -1))

    unique_flat = []
    unique_indices = []

    for i, grid in enumerate(flat_grids):
        is_unique = True
        for ref in unique_flat:
            dist = hamming(grid, ref)
            if dist <= threshold:
                is_unique = False
                break
        if is_unique:
            unique_flat.append(grid)
            unique_indices.append(i)

    unique_grids = grids[unique_indices]
    return len(unique_indices), unique_grids

def get_unique_shapes2(grids):
    """
    Identify and return the unique 38x38 binary shape grids.

    Parameters
    ----------
    grids : np.ndarray
        Array of shape (N, 38, 38), where each grid is a binary mask (0/1).

    Returns
    -------
    unique_count : int
        Number of unique shapes.
    unique_grids : np.ndarray
        Array of unique shape grids with shape (M, 38, 38), M <= N.
    """
    grids = np.asarray(grids).astype(np.uint8)
    flat_grids = grids.reshape((grids.shape[0], -1))

    seen = {}
    unique_indices = []

    for i, row in enumerate(flat_grids):
        h = hash(bytes(row))
        if h not in seen:
            seen[h] = i
            unique_indices.append(i)

    unique_grids = grids[unique_indices]
    return len(unique_indices), unique_grids


def remove_duplicates_against_reference(set_a, set_b, threshold=0.02):
    """
    Remove shapes from set_a that are similar to any shape in set_b.

    Parameters
    ----------
    set_a : np.ndarray
        Array of shape (N, 38, 38), the set to filter.
    set_b : np.ndarray
        Array of shape (M, 38, 38), the reference set to compare against.
    threshold : float
        Hamming distance threshold (0 to 1). Shapes within this distance are considered duplicates.

    Returns
    -------
    filtered_set_a : np.ndarray
        Subset of set_a that does not closely match any shape in set_b.
    """
    set_a = np.asarray(set_a).astype(np.uint8).reshape((set_a.shape[0], -1))
    set_b = np.asarray(set_b).astype(np.uint8).reshape((set_b.shape[0], -1))

    filtered_indices = []

    for i, grid_a in enumerate(set_a):
        is_duplicate = False
        for grid_b in set_b:
            if hamming(grid_a, grid_b) <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_indices.append(i)

    return set_a[filtered_indices].reshape((-1, 38, 38))


"""-------- Augumentation Shapes --------"""

def augment_opacity_maps(opacity_maps):
    """
    Generate 8 variations (original + flips + rotations) for each opacity map.

    Parameters
    ----------
    opacity_maps : np.ndarray
        Array of shape (N, H, W), where each (H, W) is a binary opacity map.

    Returns
    -------
    np.ndarray
        Array of shape (N*8, H, W) containing augmented opacity maps.
    """
    all_augmented = []

    for img in opacity_maps:
        variations = [
            img,                                      # original
            np.flipud(img),                           # vertical flip
            np.fliplr(img),                           # horizontal flip
            np.rot90(img, 1),                         # 90 deg rotation
            np.rot90(img, 2),                         # 180 deg rotation
            np.rot90(img, 3),                         # 270 deg rotation
            np.fliplr(np.rot90(img, 1)),              # 90 deg + H flip
            np.flipud(np.rot90(img, 1))               # 90 deg + V flip
        ]
        all_augmented.extend(variations)

    return np.stack(all_augmented, axis=0)
