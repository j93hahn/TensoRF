import numpy as np
import torch

import matplotlib.pyplot as plt


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
compute the weights histogram of a single ray --

given the weights of each sample along a single ray, compute the weights histogram
and return the index, weight, and xyz location of the sample at the 50th percentile
of the cumulative distribution function (CDF) of the weights histogram. we use this
value to extract the sigma value at the xyz location of the weight value at the
50th percentile of the CDF
"""
def compute_weight_histogram_of_single_ray(
    weights: np.ndarray,
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
        return idx, weights[idx], xyz_samples[idx]

    # return the maximum weight if it is >= 50% of the total sum of the weights; this
    # value must divide the weights histogram into two equal parts
    if (weights.max() / weights.sum()) >= 0.5:
        idx = np.argmax(weights)
        return idx, weights[idx], xyz_samples[idx]

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
    return idx, weights[idx], xyz_samples[idx]


"""
compute the weights histogram of multiple rays --

wrapper function that calls compute_weights_histogram_of_single_ray() for each ray in the batch

TODO: not fully vectorized over the entire batch; can be made faster
"""
def compute_weight_histograms_of_multiple_rays(weights, xyz_samples):
    # weights: N_rays x N_samples
    # xyz_samples: N_rays x N_samples x 3
    _idxs = np.zeros(weights.shape[0], dtype=np.int32)
    _weights = np.zeros(weights.shape[0], dtype=np.float32)
    _xyz_locs = np.zeros((weights.shape[0], 3), dtype=np.float32)

    # compute the weights histogram for each ray
    for ray_idx in range(weights.shape[0]):
        idx, weight, xyz_loc = compute_weight_histogram_of_single_ray(weights[ray_idx],
                                                                      xyz_samples[ray_idx])

        # store the index, weight, and xyz location of the given ray
        _idxs[ray_idx] = idx
        _weights[ray_idx] = weight
        _xyz_locs[ray_idx] = xyz_loc

    # return the indices, weights, and xyz locations of the rays
    return _idxs, _weights, _xyz_locs


if __name__ == "__main__":
    """
    Notes on data (loaded from a sample training batch of rays from TensoRF)

    weights: the weights of each sample along a single ray
    dists: the distances between each sample along a single ray
    sigma: the sigma values of each sample along a single ray
    viewdirs: the view directions of each ray (extended to match the shape of xyz_samples)
    xyz_samples: the xyz coordinates of each sample along a single ray
    rays_chunk: the origin and direction of each ray in the batch

    Ray 1788 contains the highest individual weight value
    Ray 3794 contains the highest accumulation of sigmas
    Ray 3461 contains the highest individual sigma value
    """
    weights = np.load("weights.npy")            # [4096, 1105] (N_rays, N_samples)
    dists = np.load("dists.npy")                # [4096, 1105]
    sigma = np.load("sigma.npy")                # [4096, 1105]
    viewdirs = np.load("viewdirs.npy")          # [4096, 1105, 3]
    xyz_samples = np.load("xyz_samples.npy")    # [4096, 1105, 3]
    rays_chunk = np.load("rays_chunk.npy")      # [4096, 6] (N_rays, ray_origin + ray_direction)

    # isolate a single ray to analyze here
    ray_idx = 700

    # sample 50th percentile of weights histogram for a single ray
    idx, weight, xyz_loc = compute_weight_histogram_of_single_ray(weights[ray_idx],
                                                                  xyz_samples[ray_idx],
                                                                  viz_hist=True)

    # compute 50th percentile of weights histogram for all rays in the batch
    _, _, _xyz_locs = compute_weight_histograms_of_multiple_rays(weights, xyz_samples)

    # visualize xyz locations
    sigma_vals = np.load('sigmas_new.npy')
    mask = (sigma_vals == 0)    # mask out the rays that passed through empty space
    colors = sigma_vals[~mask]
    xyz_locs = _xyz_locs[~mask]

    # visualize the xyz locations of the 50th percentile of the weights histogram
    # colorcoded with the sigma values using matplotlib
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xyz_locs[:,0], xyz_locs[:,1], xyz_locs[:,2], c=colors, cmap='magma')
    plt.show()
    plt.close()

    # visualize the xyz locations of the 50th percentile of the weights histogram
    # colorcoded with the sigma values using open3d - not supported on Mac Apple Silicon
    import open3d as o3d
    _colors = np.random.uniform(0, 1, size=(xyz_locs.shape[0], 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_xyz_locs.astype(int))
    pcd.colors = o3d.utility.Vector3dVector(_colors)
    o3d.visualization.draw_geometries([pcd])
