import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import settings

import uuid
import math
import torch
import numpy as np
import imageio.v2 as imageio

from random import randint
from nerf2vec.utils import get_rays

from torch.cuda.amp import autocast
from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, render_image
from nerf2vec import config as nerf2vec_config


@torch.no_grad()
def draw_images(decoder, embeddings, device='cuda:0', class_idx=0):

    scene_aabb = torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / nerf2vec_config.GRID_CONFIG_N_SAMPLES
    ).item()
    rays = get_rays(device)

    # WHITE BACKGROUND
    color_bkgd = torch.ones((1,3), device=device) 
    
    img_name = str(uuid.uuid4())
    plots_path = os.path.join('task_generation', f'GAN_plots_{class_idx}')
    os.makedirs(plots_path, exist_ok=True)

    for idx, emb in enumerate(embeddings):
        emb = torch.tensor(emb, device=device, dtype=torch.float32)
        emb = emb.unsqueeze(dim=0)
        with autocast():
            rgb_A, alpha, b, c, _, _ = render_image(
                            radiance_field=decoder,
                            embeddings=emb,
                            occupancy_grid=None,
                            rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                            scene_aabb=scene_aabb,
                            render_step_size=render_step_size,
                            render_bkgd=color_bkgd,
                            grid_weights=None,
                            device=device
            )

        imageio.imwrite(
            os.path.join(plots_path, f'{img_name}_{idx}.png'),
            (rgb_A.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
        )


@torch.no_grad()
def create_renderings_from_GAN_embeddings(device='cuda:0', class_idx=0, n_images=10):

    # Init nerf2vec 
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
    decoder.eval()
    decoder = decoder.to(device)

    ckpt_path = settings.GENERATION_NERF2VEC_FULL_CKPT_PATH
    print(f'loading weights: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    latent_gan_embeddings_path = settings.GENERATION_LATENT_GAN_FULL_CKPT_PATH.format(class_idx)
    embeddings = np.load(latent_gan_embeddings_path)["embeddings"]
    embeddings = torch.from_numpy(embeddings)

    for _ in range(0, n_images):
        idx = randint(0, embeddings.shape[0]-1)
        emb = embeddings[idx].unsqueeze(0).cuda()
        draw_images(decoder, emb, device, class_idx)


def main() -> None:
    # Create renderings for each class
    for class_idx in range(0, nerf2vec_config.NUM_CLASSES):
        create_renderings_from_GAN_embeddings(device=settings.DEVICE_NAME, class_idx=class_idx, n_images=10)

if __name__ == "__main__":
    main()