import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import AsinhNorm, LogNorm, Normalize


"""
compute the bin width for a histogram using the Freedman-Diaconis rule given by:
    bin_width = 2 * IQR(x) * N^(-1/3)

where IQR(x) is the interquartile range of the data x, and N is the number of
samples in the data x. the number of bins is then given by:
    num_bins = ceil((max(x) - min(x)) / bin_width)
"""
def compute_freedman_diaconis_number_bins(samples):
    # convert samples to NumPy array
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    elif isinstance(samples, list):
        samples = np.array(samples)
    elif not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy array, PyTorch tensor, or list")

    # ensure that samples is a 1D array
    assert samples.ndim == 1, "samples must be a 1D array"

    _num_samples = samples.shape[0]

    # compute the interquartile range (IQR) of the samples
    _iqr = np.percentile(samples, 75) - np.percentile(samples, 25)

    if _iqr == 0:   # if the IQR is 0, then return 1 bin
        return 1

    # compute the bin width and number of bins
    _bin_width = 2 * _iqr * np.power(_num_samples, -1/3)
    return int(np.ceil((samples.max() - samples.min()) / _bin_width))


"""
Modified OctreeRender_trilinear_fast from renderer.py --

Extracts sigmas and weight values from the forward method of the TensoRF model as opposed to the rgb_map, depth_map, and alpha values.
"""
def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    weights, sigmas, xyz_samples = [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        _, _, _, weight, sigma, xyz_sample = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        # compute the weights histogram of each ray batch-by-batch to avoid CPU memory issues
        _, _weights, _sigmas, _xyz_locs = compute_weight_histograms_of_multiple_rays_vectorized(
            weight.detach().cpu().numpy(),
            sigma.detach().cpu().numpy(),
            xyz_sample.detach().cpu().numpy(),
        )

        weights.append(_weights)
        sigmas.append(_sigmas)
        xyz_samples.append(_xyz_locs)

    return np.concatenate(weights), np.concatenate(sigmas), np.concatenate(xyz_samples)


"""
volumetric rendering equation recap --

given the weights histogram of a single ray computed via alpha compositing,
the final color of the ray is given by:
    C = \sum_i w_i * C_i

where C_i is the color of the i-th sample, w_i is the weight of the i-th sample,
and C is the final color of the ray for a single pixel.

the weight of the i-th sample is given by:
    w_i = \prod_{j=1}^{i-1} \alpha_j * (1 - \alpha_i)

where \alpha_j = 1 - \exp(-\sigma_j * d_j). here, \sigma_j is the opacity of the
j-th sample, and d_j is the interval length between the j-th and (j+1)-th sample.

compute the weights histogram of a single ray --

given the weights of each sample along a single ray, compute the weights histogram
and return the index, weight, and xyz location of the sample at the 50th percentile
of the cumulative distribution function (CDF) of the weights histogram. we use this
value to extract the sigma value at the xyz location of the weight value at the
50th percentile of the CDF
"""
def compute_weight_histogram_of_single_ray(
    weights: np.ndarray,
    sigmas: np.ndarray,
    xyz_samples: np.ndarray,
    percentile: float = 0.5,
    viz_hist: bool = False,
):
    # weights, dists, xyz_samples [N_samples]
    assert weights.shape == (xyz_samples.shape[0],), \
        "weights and xyz_samples must have the same shape"

    # if the sum of the weights is 0, the ray passed through empty space; sample a
    # random point along the ray
    if weights.sum() == 0:
        idx = np.random.randint(0, weights.shape[0])
        return 0, 0, 0, 0

    # return the maximum weight if it is >= 50% of the total sum of the weights; this
    # value must divide the weights histogram into two equal parts
    if (weights.max() / weights.sum()) >= 0.5:
        idx = np.argmax(weights)
        return idx, weights[idx], sigmas[idx], xyz_samples[idx]

    # normalize the weights to sum 1 and compute its cumulative distribution function
    _weights = weights / weights.sum()
    _weights = np.cumsum(_weights)

    # visualize the weights CDF if specified
    if viz_hist:
        plt.figure()
        plt.grid(True)
        plt.plot(_weights)
        plt.xlabel("Weight Sample Along Ray [Origin to Endpoint]")
        plt.ylabel("CDF")
        plt.title("CDF of Weights Histogram of Ray")
        plt.savefig("weights_hist.png", dpi=300)
        plt.show()
        plt.close()

    # extract the first weight value at the specified percentile of the CDF
    idx = np.where(_weights >= percentile)[0][0]

    # return the weight value at the 50th percentile of the CDF
    return idx, weights[idx], sigmas[idx], xyz_samples[idx]


"""
compute the weights histogram of multiple rays --

non-vectorized wrapper function that calls compute_weights_histogram_of_single_ray() for each ray in the batch
"""
def compute_weight_histograms_of_multiple_rays(weights, sigmas, xyz_samples):
    # weights: N_rays x N_samples
    # xyz_samples: N_rays x N_samples x 3
    _idxs = np.zeros(weights.shape[0], dtype=np.int32)
    _weights = np.zeros(weights.shape[0], dtype=np.float32)
    _sigmas = np.zeros(weights.shape[0], dtype=np.float32)
    _xyz_locs = np.zeros((weights.shape[0], 3), dtype=np.float32)

    # compute the weights histogram for each ray
    for ray_idx in range(weights.shape[0]):
        idx, weight, sigma, xyz_loc = compute_weight_histogram_of_single_ray(weights[ray_idx],
                                                                             sigmas[ray_idx],
                                                                             xyz_samples[ray_idx])

        # store the index, weight, and xyz location of the given ray
        _idxs[ray_idx] = idx
        _weights[ray_idx] = weight
        _sigmas[ray_idx] = sigma
        _xyz_locs[ray_idx] = xyz_loc

    # return the indices, weights, and xyz locations of the rays
    return _idxs, _weights, _sigmas, _xyz_locs


"""
compute the weights histogram of multiple rays --

vectorized wrapper function that computes the weights histogram of every ray in the batch simultaneously
"""
def compute_weight_histograms_of_multiple_rays_vectorized(
    weights: np.ndarray,
    sigmas: np.ndarray,
    xyz_samples: np.ndarray,
    percentile: float = 0.5,
):
    # weights: N_rays x N_samples
    # xyz_samples: N_rays x N_samples x 3
    _idxs = np.zeros(weights.shape[0], dtype=np.int32)
    _weights = np.zeros(weights.shape[0], dtype=np.float32)
    _sigmas = np.zeros(weights.shape[0], dtype=np.float32)
    _xyz_locs = np.zeros((weights.shape[0], 3), dtype=np.float32)

    # mask will store the indices of the rays that have not yet been processed
    # as false; once a ray has been processed, its index will be set to true
    mask = np.zeros(weights.shape[0], dtype=bool)
    _weights_sum = weights.sum(axis=-1)

    # if the sum of the weights is 0, the ray passed through empty space; apply a
    # mask to that ray as it will not contribute to the final image
    if np.any(_weights_sum == 0):
        mask[np.where(_weights_sum == 0)[0]] = True
        if mask.sum() == weights.shape[0]:
            return _idxs, _weights, _sigmas, _xyz_locs

    # return the maximum weight if it is >= 50% of the total sum of the weights; this
    # value must divide the weights histogram into two equal parts
    np.seterr(divide='ignore', invalid='ignore')    # ignore divide by zero warnings
    if np.any((weights[~mask].max(axis=-1) / _weights_sum[~mask]) >= 0.5):
        _wmax = np.nan_to_num(weights.max(axis=-1) / _weights_sum, nan=0.0)
        _wmax = (_wmax >= 0.5) & (~mask)
        _idxs[_wmax] = weights.argmax(axis=-1)[_wmax]
        _weights[_wmax] = weights[_wmax, _idxs[_wmax]]
        _sigmas[_wmax] = sigmas[_wmax, _idxs[_wmax]]
        _xyz_locs[_wmax] = xyz_samples[_wmax, _idxs[_wmax]]

        # apply a mask to the rays that have been processed
        mask[_wmax] = True
        if mask.sum() == weights.shape[0]:
            return _idxs, _weights, _sigmas, _xyz_locs

    # normalize the weights of each ray to sum 1 and compute its cumulative distribution function
    weights_cum = np.nan_to_num(weights / _weights_sum[..., None], nan=0.0)
    weights_cum = np.cumsum(weights_cum, axis=-1)

    # extract the first weight value at the specified percentile of the CDF for each ray
    _wpercentile = np.apply_along_axis(lambda x: np.where(x >= percentile)[0][0], axis=-1, arr=weights_cum[~mask])

    # store the index, weight, and xyz location of the given ray
    _idxs[~mask] = _wpercentile
    _weights[~mask] = weights[~mask, _idxs[~mask]]
    _sigmas[~mask] = sigmas[~mask, _idxs[~mask]]
    _xyz_locs[~mask] = xyz_samples[~mask, _idxs[~mask]]

    # turn divide by zero warnings back on
    np.seterr(divide='warn', invalid='warn')

    # return the indices, weights, and xyz locations of the rays
    return _idxs, _weights, _sigmas, _xyz_locs


def create_single_sigma_viz(_sigmas, pose):
    fig = plt.figure(figsize=(5, 6))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 1),
        cbar_location="right", cbar_mode="edge", cbar_size="7%", cbar_pad=0.15,
    )

    lower = 1e-2
    upper = 1e2

    _sigmas[_sigmas < lower] = lower
    _sigmas[_sigmas > upper] = upper

    # (800,800) for blender; (756,1008) for LLFF
    h = grid[0].imshow(_sigmas.reshape(800,800), cmap='viridis', norm=LogNorm(lower, upper))
    grid[0].set_title(f'Sigma Visualizations at Pose {pose}')
    grid[0].get_xaxis().set_visible(False)
    grid[0].get_yaxis().set_visible(False)

    plt.colorbar(h, cax=grid.cbar_axes[0])
    plt.savefig(f'sigmas_pose{pose}.png', dpi=300)
    plt.close()
