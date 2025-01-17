import os
import sys
from pathlib import Path

import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, LogNorm, Normalize
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from dataLoader.ray_utils import get_rays, ndc_rays_blender
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from opt import config_parser
from utils import *
from viz_utils import OctreeRender_trilinear_fast, create_single_sigma_viz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast


@torch.no_grad()
def generate_sigmas(
    test_dataset,           # test dataset
    tensorf,                # tensor field model
    args,                   # command line arguments
    renderer,               # octree renderer
    N_vis=5,                # number of images to visualize
    N_samples=-1,           # number of samples to use for each ray
    white_bg=False,         # white background
    ndc_ray=False,          # use ndc rays
    device='cuda',          # device to use
):
    # clear the tqdm progress bar
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    writer = imageio.get_writer(f'sigmas_{Path(os.getcwd()).parts[-1]}.mp4', fps=20)

    if args.save_xyz_loc:
        print("\nSaving xyz locations for future samplings")

    if args.load_xyz_loc:
        if args.load_xyz_loc_path is None:
            raise ValueError("Please specify the path to load the xyz locations from")
        print("\nLoading non-normalized xyz locations from previous samplings")

    # render the sigma test pose
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        if args.save_xyz_loc:
            rays = samples.view(-1,samples.shape[-1])
            _, _sigmas, _xyz_locs = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                             ndc_ray=ndc_ray, white_bg=white_bg, device=device)

            # save the xyz locations for future samplings (for the same pose)
            np.save(f'xyz_locs_pose{idx}.npy', _xyz_locs)
            np.save(f'sigmas_pose{idx}.npy', _sigmas)

            create_single_sigma_viz(_sigmas, idx)
            writer.append_data(imageio.imread(f'sigmas_pose{idx}.png'))

        elif args.load_xyz_loc:
            _xyzs = Path(args.load_xyz_loc_path) / f'xyz_locs_pose{idx}.npy'
            xyzs = torch.from_numpy(np.load(_xyzs)).to(device)

            _sigmas = tensorf.compute_sigma(xyzs)
            np.save(f'sigmas_pose{idx}.npy', _sigmas)

            create_single_sigma_viz(_sigmas, idx)
            writer.append_data(imageio.imread(f'sigmas_pose{idx}.png'))

        else:
            raise ValueError("Please specify either save_xyz_loc or load_xyz_loc")

    writer.close()


@torch.no_grad()
def render_sigmas_from_test_pose(args):
    # extract dataset first
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    # load the checkpointed model
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # generate the sigma values
    generate_sigmas(test_dataset, tensorf, args, renderer, N_vis=-1, N_samples=-1,
                    white_bg=white_bg, ndc_ray=ndc_ray, device=device)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    render_sigmas_from_test_pose(args)
