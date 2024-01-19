import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import settings

import json
import math
import time
import wandb
import random
import datetime

from nerfacc import OccupancyGrid
from nerf2vec.utils import get_latest_checkpoints_path, get_mlp_params_as_matrix

import torch
import numpy as np
import torch.nn.functional as F

from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from nerf.intant_ngp import NGPradianceField

from pathlib import Path
from typing import Any, Dict

from nerf2vec import config as nerf2vec_config
from nerf.loader_gt import NeRFLoaderGT
from nerf.utils import Rays, render_image, render_image_GT

class NeRFDataset(Dataset):
    def __init__(self, split_json: str, device: str) -> None:
        super().__init__()

        with open(split_json) as file:
            self.nerf_paths = json.load(file)
        
        assert isinstance(self.nerf_paths, list), 'The json file provided is not a list.'

        self.device = device

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:

        data_dir = self.nerf_paths[index]

        nerf_loader = NeRFLoaderGT(
            data_dir=data_dir,
            num_rays=nerf2vec_config.NUM_RAYS,
            device=self.device)

        nerf_loader.training = True
        data = nerf_loader[0]  # NOTE: The index has not any relevance when training is True
        color_bkgd = data["color_bkgd"]
        rays = data["rays"]
        train_nerf = {
            'rays': rays,
            'color_bkgd': color_bkgd
        }

        nerf_loader.training = False
        
        # Getting just the first image in the dataset for performance reasons. In the future, we could use more elements.
        test_data = nerf_loader[0]  
        test_color_bkgd = test_data["color_bkgd"]
        test_rays = test_data["rays"]
        test_nerf = {
            'rays': test_rays,
            'color_bkgd': test_color_bkgd
        }

        mlp_weights = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))  # The NeRF weights obtained from NerfAcc
        mlp_matrix = get_mlp_params_as_matrix(mlp_weights['mlp_base.params'])  # The NeRF weights with proper padding

        grid_weights_path = os.path.join(data_dir, 'grid.pth')  
        grid_weights = torch.load(grid_weights_path, map_location=self.device)
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()
        n_total_cells = nerf2vec_config.GRID_NUMBER_OF_CELLS
        grid_weights['occs'] = torch.empty([n_total_cells]) 
        
        N = nerf2vec_config.GRID_BACKGROUND_CELLS_TO_SAMPLE
        background_indices, n_true_coordinates = self._sample_unoccupied_cells(N, grid_weights['_binary'], data_dir, n_total_cells)

        return train_nerf, test_nerf, mlp_weights, mlp_matrix, grid_weights, data_dir, background_indices, n_true_coordinates
    
    def _sample_unoccupied_cells(self, n: int, binary: torch.Tensor, data_dir, n_total_cells: int) -> torch.Tensor:
        
        # 0 -> PERMUTAION
        # 1 -> BETAVARIATE
        # 2 -> UNIFORM SAMPLE WITHOUT REPLACEMENT
        SAMPLE_MODE = 2

        # This method was implemented to test different sampling modes, and the best results were obtained by
        # using the SAMPLE_MODE = 2. The other sampling methods have been left here as a reference for future work.

        zero_indices = torch.nonzero(binary.flatten() == 0)[:, 0]
        n_one_indices = n_total_cells - len(zero_indices)

        if len(zero_indices) < n:
            print(f'ERROR: {len(zero_indices)} - {data_dir}')

        # PERMUTATION
        if SAMPLE_MODE == 0:
            randomized_indices = zero_indices[torch.randperm(zero_indices.size(0))][:n]

        # BETAVARIATE
        elif SAMPLE_MODE == 1:
            alpha = 4
            beta = 4

            probabilities = [random.betavariate(alpha, beta) for _ in range(len(zero_indices))]

            # Normalize the probabilities to sum up to 1
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]

            # Perform weighted random sampling to get n indices
            randomized_indices = random.choices(range(0, len(zero_indices)), probabilities, k=n)

        # UNIFORM SAMPLE WITHOUT REPLACEMENT
        elif SAMPLE_MODE == 2:
            randomized_indices = random.sample(range(0, len(zero_indices)), n)
        
        randomized_indices = zero_indices[randomized_indices]
        return randomized_indices, n_one_indices
        
    
class Nerf2vecTrainer:
    def __init__(self, device='cuda:0') -> None:

        self.device = device

        train_dset = NeRFDataset(settings.TRAIN_DSET_JSON, device='cpu') 
        self.train_loader = DataLoader(
            train_dset,
            batch_size=nerf2vec_config.BATCH_SIZE,
            shuffle=True, 
            num_workers=8, 
            persistent_workers=False, 
            prefetch_factor=2
        )

        val_dset = NeRFDataset(settings.VAL_DSET_JSON, device='cpu')   
        self.val_loader = DataLoader(
            val_dset,
            batch_size=nerf2vec_config.BATCH_SIZE,
            shuffle=False,
            num_workers=8, 
            persistent_workers=False
        )
        self.val_loader_shuffled = DataLoader(
            val_dset,
            batch_size=nerf2vec_config.BATCH_SIZE,
            shuffle=True,
            num_workers=8, 
            persistent_workers=False
        )
        
        encoder = Encoder(
            nerf2vec_config.MLP_UNITS,
            nerf2vec_config.ENCODER_HIDDEN_DIM,
            nerf2vec_config.ENCODER_EMBEDDING_DIM
        )
        self.encoder = encoder.to(self.device)

        decoder = ImplicitDecoder(
            embed_dim=nerf2vec_config.ENCODER_EMBEDDING_DIM,
            in_dim=nerf2vec_config.DECODER_INPUT_DIM,
            hidden_dim=nerf2vec_config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=nerf2vec_config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=nerf2vec_config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=nerf2vec_config.DECODER_OUT_DIM,
            encoding_conf=nerf2vec_config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=self.device)
        )
        self.decoder = decoder.to(self.device)

        occupancy_grid = OccupancyGrid(
            roi_aabb=nerf2vec_config.GRID_AABB,
            resolution=nerf2vec_config.GRID_RESOLUTION,
            contraction_type=nerf2vec_config.GRID_CONTRACTION_TYPE,
        )
        self.occupancy_grid = occupancy_grid.to(self.device)
        self.occupancy_grid.eval()

        self.ngp_mlp = NGPradianceField(**nerf2vec_config.INSTANT_NGP_MLP_CONF).to(device)
        self.ngp_mlp.eval()

        self.scene_aabb = torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=self.device)
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            * math.sqrt(3)
            / nerf2vec_config.GRID_CONFIG_N_SAMPLES
        ).item()

        lr = nerf2vec_config.LR
        wd = nerf2vec_config.WD
        params = list(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        
        self.optimizer = AdamW(params, lr, weight_decay=wd)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = float("-inf")

        self.ckpts_path = Path(settings.NERF2VEC_CKPTS_PATH)
        self.all_ckpts_path = Path(settings.NERF2VEC_ALL_CKPTS_PATH)

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.all_ckpts_path.mkdir(parents=True, exist_ok=True)

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def train(self):

        self.config_wandb()

        num_epochs = nerf2vec_config.NUM_EPOCHS
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, num_epochs):

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()
            
            print(f'Epoch {epoch} started...')          
            epoch_start = time.time()
            batch_start = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                train_nerf, _, mlp_weights, mlp_matrix, grid_weights, _, background_indices, _ = batch

                rays = train_nerf['rays']
                color_bkgds = train_nerf['color_bkgd']
                color_bkgds = color_bkgds[0][None].expand(len(mlp_matrix), -1)

                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                color_bkgds = color_bkgds.cuda()
                mlp_matrix = mlp_matrix.cuda()

                # Enable autocast for mixed precision
                with autocast():
                    pixels, alpha, filtered_rays = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=mlp_weights,
                            device=self.device)
                    pixels = pixels * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)

                    embeddings = self.encoder(mlp_matrix)
                    
                    rgb, _, _, _,  bg_rgb_pred, bg_rgb_label = render_image(
                        self.decoder,
                        embeddings,
                        self.occupancy_grid,
                        filtered_rays,
                        self.scene_aabb,
                        render_step_size=self.render_step_size,
                        render_bkgd=color_bkgds,
                        grid_weights=grid_weights,
                        background_indices=background_indices,
                        max_foreground_coordinates=nerf2vec_config.MAX_FOREGROUND_COORDINATES,
                        max_background_coordinates=nerf2vec_config.MAX_BACKGROUND_COORDINATES,
                        device=self.device
                    )
                    
                    fg_loss = F.smooth_l1_loss(rgb, pixels) * nerf2vec_config.FG_WEIGHT
                    bg_loss = F.smooth_l1_loss(bg_rgb_pred, bg_rgb_label) * nerf2vec_config.BG_WEIGHT
                    loss = fg_loss + bg_loss
                    
                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()

                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                
                self.global_step += 1

                batch_end = time.time()
                if batch_idx % 1000 == 0:
                    print(f'Completed {batch_idx} batches in {batch_end-batch_start}s')

            if (epoch > 0 and epoch % 10 == 0) or epoch == num_epochs - 1:
                self.val(split='train')
                self.val(split='validation')

                self.plot(split='train')
                self.plot(split='validation')

            if epoch % 50 == 0:
                self.save_ckpt(all=True)
            
            self.save_ckpt()
            
            epoch_end = time.time()
            print(f'Epoch {epoch} completed in {epoch_end-epoch_start}s')        
    
    @torch.no_grad()
    def val(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()
        self.decoder.eval()

        psnrs = []
        psnrs_bg = []
        idx = 0

        print(f'Validating on {split} set')

        for batch_idx, batch in enumerate(loader):

            train_nerf, _, mlp_weights, mlp_matrix, grid_weights, _, background_indices, _ = batch
            rays = train_nerf['rays']
            color_bkgds = train_nerf['color_bkgd']
            color_bkgds = color_bkgds[0].unsqueeze(0).expand(len(mlp_matrix), -1)
            
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
            color_bkgds = color_bkgds.cuda()
            mlp_matrix = mlp_matrix.cuda()
            with autocast():
                pixels, alpha, filtered_rays = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=mlp_weights,
                            device=self.device)
                pixels = pixels * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)
                
                embeddings = self.encoder(mlp_matrix)
                
                rgb, _, _, _, bg_rgb_pred, bg_rgb_label = render_image(
                    self.decoder,
                    embeddings,
                    self.occupancy_grid,
                    filtered_rays,
                    self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds,
                    grid_weights=grid_weights,
                    background_indices=background_indices,
                    max_foreground_coordinates=nerf2vec_config.MAX_FOREGROUND_COORDINATES,
                    max_background_coordinates=nerf2vec_config.MAX_BACKGROUND_COORDINATES,
                    device=self.device
                )
                
                fg_mse = F.mse_loss(rgb, pixels) * nerf2vec_config.FG_WEIGHT
                bg_mse = F.mse_loss(bg_rgb_pred, bg_rgb_label) * nerf2vec_config.BG_WEIGHT

                mse_bg = fg_mse + bg_mse
                mse = F.mse_loss(rgb, pixels)
            
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())

            psnr_bg = -10.0 * torch.log(mse_bg) / np.log(10.0)
            psnrs_bg.append(psnr_bg.item())

            if idx > 99:
                break
            idx+=1
        
        mean_psnr = sum(psnrs) / len(psnrs)
        mean_psnr_bg = sum(psnrs_bg) / len(psnrs_bg)

        self.logfn({f'{split}/PSNR': mean_psnr})
        self.logfn({f'{split}/PSNR_BG': mean_psnr_bg})
        
        if split == 'validation' and mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save_ckpt(best=True)
    
    @torch.no_grad()
    def plot(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader_shuffled

        print('Plot started...')

        self.encoder.eval()
        self.decoder.eval()

        loader_iter = iter(loader)
        _, test_nerf, mlp_weights, mlp_matrix, grid_weights, _, _, _ = next(loader_iter)
        
        rays = test_nerf['rays']
        color_bkgds = test_nerf['color_bkgd']
        
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

        color_bkgds = color_bkgds.cuda()
        mlp_matrix = mlp_matrix.cuda()
        
        with autocast():
            pixels, alpha, _ = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=mlp_weights,
                            device=self.device,
                            training=False)
            pixels = pixels * alpha + color_bkgds.unsqueeze(1).unsqueeze(1) * (1.0 - alpha)
        
            embeddings = self.encoder(mlp_matrix)

            for idx in range(len(mlp_matrix)):
                
                curr_grid_weights = {
                    '_roi_aabb': [grid_weights['_roi_aabb'][idx]],
                    '_binary': [grid_weights['_binary'][idx]],
                    'resolution': [grid_weights['resolution'][idx]],
                    'occs': [grid_weights['occs'][idx]],
                }
        
                rgb, _, _, _, _, _ = render_image(
                    radiance_field=self.decoder,
                    embeddings=embeddings[idx].unsqueeze(dim=0),
                    occupancy_grid=self.occupancy_grid,
                    rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                    scene_aabb=self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                    grid_weights=curr_grid_weights,
                    device=self.device
                )
                
                rgb_A, alpha, _, _, _, _ = render_image(
                                radiance_field=self.decoder,
                                embeddings=embeddings[idx].unsqueeze(dim=0),
                                occupancy_grid=None,
                                rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                                scene_aabb=self.scene_aabb,
                                render_step_size=self.render_step_size,
                                render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                                grid_weights=None,
                                device=self.device
                )
                
                gt_image = wandb.Image((pixels.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))
                pred_image_grid = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
                pred_image_no_grid = wandb.Image((rgb_A.to('cpu').detach().numpy() * 255).astype(np.uint8))

                self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image_grid, pred_image_no_grid]})
                 
    def save_ckpt(self, best: bool = False, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_psnr": self.best_psnr,
        }

        if all:
            ckpt_path = self.all_ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        else:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "best" not in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()

            ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)
    
    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_path = get_latest_checkpoints_path(self.ckpts_path)
            print(f'loading weights: {ckpt_path}')
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_psnr = ckpt["best_psnr"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
    
    def config_wandb(self):
        wandb.init(
            entity='entity',
            project='nerf2vec',
            name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=nerf2vec_config.WANDB_CONFIG
        )

def main():
    nerf2vec = Nerf2vecTrainer(device=settings.DEVICE_NAME)
    nerf2vec.train()

if __name__ == "__main__":
    main()