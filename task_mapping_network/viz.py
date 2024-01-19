import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import settings

import math
from random import randint
import shutil
import uuid

import tqdm

from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
from hesiod import hcfg, hmain
from torch import Tensor
from torch.utils.data import Dataset

from task_mapping_network.inr2vec.models.idecoder import ImplicitDecoder as INRDecoder
from task_mapping_network.inr2vec.models.transfer import Transfer
from models.idecoder import ImplicitDecoder as NeRFDecoder
from nerf.utils import Rays, render_image, render_image_GT
from nerf2vec.utils import generate_rays, pose_spherical

from nerfacc import OccupancyGrid
from nerf.intant_ngp import NGPradianceField
from nerf2vec import config as nerf2vec_config

from torch.cuda.amp import autocast

import imageio.v2 as imageio


class InrEmbeddingDataset(Dataset):
    def __init__(self, nerfs_root: Path, inrs_root: Path, split: str) -> None:
        super().__init__()

        nerfs_root = nerfs_root / split
        inrs_root = inrs_root / split

        self.nerf_item_paths = sorted(nerfs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.inr_item_paths = sorted(inrs_root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.nerf_item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with h5py.File(self.inr_item_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            embedding_pcd = np.array(f.get("embedding"))
            embedding_pcd = torch.from_numpy(embedding_pcd)
            uuid_pcd = f.get("uuid")[()].decode()

        
        with h5py.File(self.nerf_item_paths[index], "r") as f:
            nerf_data_dir = f.get("data_dir")[()].decode()
            
            embedding_nerf = np.array(f.get("embedding"))
            embedding_nerf = torch.from_numpy(embedding_nerf)
            uuid_nerf = f.get("uuid")[()].decode()
        
        assert uuid_nerf == uuid_pcd, "UUID ERROR"

        return embedding_nerf, nerf_data_dir, pcd, embedding_pcd, uuid_pcd


@hmain(
    base_cfg_dir="cfg/bases",
    run_cfg_file="task_mapping_network/cfg/completion.yaml",
    parse_cmd_line=False,
    create_out_dir=False,
    out_dir_root="task_mapping_network/logs"
)
@torch.no_grad()
def main() -> None:

    inrs_dset_root = Path(hcfg("inrs_dset_root", str))  
    nerfs_dset_root = Path(hcfg("nerfs_dset_root", str))
    
    split_name = 'train'
    dset_root = hcfg(f"{split_name}_split", str)
    dset = InrEmbeddingDataset(nerfs_dset_root, inrs_dset_root, dset_root)

    embedding_dim = hcfg("embedding_dim", int)
    num_layers = hcfg("num_layers_transfer", int)
    transfer = Transfer(embedding_dim, num_layers)
    transfer = transfer.cuda()

    # ####################
    # INR DECODER
    # ####################
    inr_decoder_cfg = hcfg("inr_decoder", Dict[str, Any])
    inr_decoder = INRDecoder(
            embedding_dim,
            inr_decoder_cfg["input_dim"],
            inr_decoder_cfg["hidden_dim"],
            inr_decoder_cfg["num_hidden_layers_before_skip"],
            inr_decoder_cfg["num_hidden_layers_after_skip"],
            inr_decoder_cfg["out_dim"],
    )
    inr_decoder_ckpt_path = hcfg("inr2vec_decoder_ckpt_path", str) 
    inr_decoder_ckpt = torch.load(inr_decoder_ckpt_path, map_location="cpu")
    inr_decoder.load_state_dict(inr_decoder_ckpt["decoder"])
    inr_decoder = inr_decoder.cuda()
    inr_decoder.eval()

    # ####################
    # NeRF DECODER
    # ####################
    nerf_decoder_cfg = hcfg("nerf_decoder", Dict[str, Any])

    INSTANT_NGP_ENCODING_CONF = {
            "otype": "Frequency",
            "n_frequencies": 24
    }
    GRID_AABB = [-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]

    nerf_decoder = NeRFDecoder(
            embed_dim=embedding_dim,
            in_dim=nerf_decoder_cfg["input_dim"],
            hidden_dim=nerf_decoder_cfg["hidden_dim"],
            num_hidden_layers_before_skip=nerf_decoder_cfg["num_hidden_layers_before_skip"],
            num_hidden_layers_after_skip=nerf_decoder_cfg["num_hidden_layers_after_skip"],
            out_dim=nerf_decoder_cfg["out_dim"],
            encoding_conf=INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(GRID_AABB, dtype=torch.float32).cuda()
    )
    nerf_decoder.eval()
    nerf_decoder = nerf_decoder.cuda()
    ckpt_path = hcfg("nerf2vec_decoder_ckpt_path", str)
    print(f'loading nerf2vec weights: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    nerf_decoder.load_state_dict(ckpt["decoder"])

    # ####################
    # NerfAcc 
    # ####################
    device = settings.DEVICE_NAME

    occupancy_grid = OccupancyGrid(
            roi_aabb=nerf2vec_config.GRID_AABB,
            resolution=nerf2vec_config.GRID_RESOLUTION,
            contraction_type=nerf2vec_config.GRID_CONTRACTION_TYPE,
    )
    occupancy_grid = occupancy_grid.to(device)
    occupancy_grid.eval()

    ngp_mlp = NGPradianceField(**nerf2vec_config.INSTANT_NGP_MLP_CONF).to(device)
    ngp_mlp.eval()

    scene_aabb = torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / nerf2vec_config.GRID_CONFIG_N_SAMPLES
    ).item()
    
    ckpt_path = hcfg("completion_ckpt_path", str)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    embedding_dim = hcfg("embedding_dim", int)
    num_layers = hcfg("num_layers_transfer", int)
    transfer = Transfer(embedding_dim, num_layers)
    transfer.load_state_dict(ckpt["net"])
    transfer = transfer.cuda()
    transfer.eval()
    
    n_generated_items = 0
    while True:
        idx = randint(0, len(dset) - 1)
        print("Index:", idx)

        
        embedding_nerf, nerf_data_dir, pcd, embedding_pcd, uuid_pcd = dset[idx]
        
        nerf_path = os.path.join(nerf_data_dir, nerf2vec_config.NERF_WEIGHTS_FILE_NAME)
        nerf = torch.load(nerf_path, map_location=torch.device('cpu'))  
        nerf['mlp_base.params'] = [nerf['mlp_base.params']]

        grid_weights_path = os.path.join(nerf_data_dir, 'grid.pth')  
        grid = torch.load(grid_weights_path, map_location='cpu')
        grid['_binary'] = grid['_binary'].to_dense()
        n_total_cells = nerf2vec_config.GRID_NUMBER_OF_CELLS
        grid['occs'] = torch.empty([n_total_cells]) 
        grid = {
            '_roi_aabb': [grid['_roi_aabb']],
            '_binary': [grid['_binary']],
            'resolution': [grid['resolution']],
            'occs': [grid['occs']],
        }
        
        embedding_pcd = embedding_pcd.unsqueeze(0).cuda()
        embedding_nerf = embedding_nerf.unsqueeze(0).cuda()

        with torch.no_grad():
            embeddings_transfer = transfer(embedding_pcd)
        
        # TODO: comment these parameters!
        # NeRF rendering parameters
        width = 224
        height = 224
        camera_angle_x = 0.8575560450553894 # Parameter taken from traned NeRFs
        focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

        max_images = 3
        array = [-30] * max_images

        # White background
        color_bkgds = torch.ones((1,3)) 
        color_bkgds = color_bkgds.cuda()

        plots_path = os.path.join('task_mapping_network', 'completion_plots')
        renderings_root_path = os.path.join(plots_path, str(uuid.uuid4()))   
        os.makedirs(renderings_root_path, exist_ok=True) 

        for n_img, theta in tqdm.tqdm(enumerate(np.linspace(0.0, 360.0, max_images, endpoint=False))):
            c2w = pose_spherical(torch.tensor(theta), torch.tensor(array[n_img]), torch.tensor(1.5))
            c2w = c2w.to(device)
            rays = generate_rays(device, width, height, focal_length, c2w)
            with autocast():
                rgb_pred, _, _, _, _, _ = render_image(
                    radiance_field=nerf_decoder,
                    embeddings=embeddings_transfer,
                    occupancy_grid=None,
                    rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                    scene_aabb=scene_aabb,
                    render_step_size=render_step_size,
                    render_bkgd=color_bkgds.unsqueeze(dim=0),
                    grid_weights=None,
                    device=device
                )
            
                rgb_gt, _, _ = render_image_GT(
                    radiance_field=ngp_mlp, 
                    occupancy_grid=occupancy_grid, 
                    rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)), 
                    scene_aabb=scene_aabb, 
                    render_step_size=render_step_size,
                    color_bkgds=color_bkgds.unsqueeze(dim=0),
                    grid_weights=grid,
                    ngp_mlp_weights=nerf,
                    device=device,
                    training=False
                )
                            
            
            full_path = os.path.join(renderings_root_path, f'nerf_pred_{n_img}.png')        
            imageio.imwrite(
                full_path,
                (rgb_pred.cpu().detach().numpy()[0] * 255).astype(np.uint8)
            )

            full_path = os.path.join(renderings_root_path, f'nerf_gt_{n_img}.png')        
            imageio.imwrite(
                full_path,
                (rgb_gt.cpu().detach().numpy()[0] * 255).astype(np.uint8)
            )
            
        pcd_root_path = hcfg("pcd_root", str)
        nerf_path = Path(nerf_data_dir)
        pcd_source_path = os.path.join(pcd_root_path, nerf_path.parts[-2], split_name, f'{nerf_path.parts[-1]}.ply')
        pcd_destination_path = os.path.join(renderings_root_path, f'{nerf_path.parts[-2]}_{nerf_path.parts[-1]}.ply')
        shutil.copy(pcd_source_path, pcd_destination_path)

        n_generated_items += 1

        print(f'generated: {n_generated_items} elements')

        if n_generated_items >= 200:
            break

if __name__ == "__main__":
    main()