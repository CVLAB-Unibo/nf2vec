import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import settings

import math
import uuid
import torch
import numpy as np
import imageio.v2 as imageio

from pathlib import Path
from random import randint
from nerf2vec import config as nerf2vec_config
from nerf2vec.train_nerf2vec import NeRFDataset
from nerf2vec.utils import get_class_label_from_nerf_root_path
from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, render_image
from torch.cuda.amp import autocast


def draw_images(
        rays, 
        color_bkgds, 
        embeddings, 
        decoder,
        scene_aabb,
        render_step_size,
        curr_folder_path,
        device):

    for idx in range(len(embeddings)):
        with autocast():
            rgb, _, _, _, _, _ = render_image(
                    radiance_field=decoder,
                    embeddings=embeddings[idx].unsqueeze(dim=0),
                    occupancy_grid=None,
                    rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                    scene_aabb=scene_aabb,
                    render_step_size=render_step_size,
                    render_bkgd=color_bkgds.unsqueeze(dim=0),
                    grid_weights=None,
                    device=device
                )
        
        img_name = f'{idx}.png'
        full_path = os.path.join(curr_folder_path, img_name)
        
        imageio.imwrite(
            full_path,
            (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
        )


@torch.no_grad()
def do_interpolation(device = 'cuda:0', split = nerf2vec_config.TRAIN_SPLIT):
    scene_aabb = torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / nerf2vec_config.GRID_CONFIG_N_SAMPLES
    ).item()


    ckpts_path = Path(settings.NERF2VEC_CKPTS_PATH)
    ckpt_paths = [p for p in ckpts_path.glob("*.pt") if "best" not in p.name]
    ckpt_path = ckpt_paths[0]
    ckpt = torch.load(ckpt_path)
    
    print(f'loaded weights: {ckpt_path}')

    encoder = Encoder(
                nerf2vec_config.MLP_UNITS,
                nerf2vec_config.ENCODER_HIDDEN_DIM,
                nerf2vec_config.ENCODER_EMBEDDING_DIM
                )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    decoder = ImplicitDecoder(
            embed_dim=nerf2vec_config.ENCODER_EMBEDDING_DIM,
            in_dim=nerf2vec_config.DECODER_INPUT_DIM,
            hidden_dim=nerf2vec_config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=nerf2vec_config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=nerf2vec_config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=nerf2vec_config.DECODER_OUT_DIM,
            encoding_conf=nerf2vec_config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=device)
        )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    dset_json_path = get_dset_json_path(split)
    dset = NeRFDataset(dset_json_path, device='cpu')  

    n_images = 0
    max_images = 100
    
    while n_images < max_images:
        idx_A = randint(0, len(dset) - 1)
        _, test_nerf_A, matrices_unflattened_A, matrices_flattened_A, _, data_dir_A, _, _ = dset[idx_A]
        class_id_A = get_class_label_from_nerf_root_path(data_dir_A)

        # Ignore augmented samples
        if is_nerf_augmented(data_dir_A):
            continue
        matrices_unflattened_A = matrices_unflattened_A['mlp_base.params']

        class_id_B = -1
        while class_id_B != class_id_A:
            idx_B = randint(0, len(dset) - 1)
            _, _, matrices_unflattened_B, matrices_flattened_B, _, data_dir_B, _, _ = dset[idx_B]
            class_id_B = get_class_label_from_nerf_root_path(data_dir_B)
        
        if is_nerf_augmented(data_dir_B):
            continue
        matrices_unflattened_B = matrices_unflattened_B['mlp_base.params']
        
        print(f'Progress: {n_images}/{max_images}')
        
        matrices_flattened_A = matrices_flattened_A.cuda().unsqueeze(0)
        matrices_flattened_B = matrices_flattened_B.cuda().unsqueeze(0)

        with autocast():
            embedding_A = encoder(matrices_flattened_A).squeeze(0)  
            embedding_B = encoder(matrices_flattened_B).squeeze(0)  
        

        embeddings = [embedding_A]
        for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            emb_interp = (1 - gamma) * embedding_A + gamma * embedding_B
            embeddings.append(emb_interp)
        embeddings.append(embedding_B)

        curr_folder_path = os.path.join('task_interp_and_retrieval', f'interp_plots_{split}', str(uuid.uuid4()))    
        os.makedirs(curr_folder_path, exist_ok=True)

        rays = test_nerf_A['rays']
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

        # WHITE BACKGROUND
        color_bkgds = torch.ones(test_nerf_A['color_bkgd'].shape)
        color_bkgds = color_bkgds.cuda()

        # Interpolation
        draw_images(
            rays, 
            color_bkgds, 
            embeddings,
            decoder,
            scene_aabb,
            render_step_size,
            curr_folder_path,
            device
        )
        
        n_images += 1

def get_dset_json_path(split):
    dset_json_path = settings.TRAIN_DSET_JSON

    if split == nerf2vec_config.VAL_SPLIT:
        dset_json_path = settings.VAL_DSET_JSON
    else:
        dset_json_path = settings.TEST_DSET_JSON
    
    
    return dset_json_path

def is_nerf_augmented(data_dir):
    return "_A1" in data_dir or "_A2" in data_dir

def main() -> None:
    do_interpolation(device=settings.DEVICE_NAME, split=nerf2vec_config.TRAIN_SPLIT)

if __name__ == "__main__":
    main()