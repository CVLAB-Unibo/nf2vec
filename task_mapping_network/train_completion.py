import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import settings

import math

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hesiod import get_cfg_copy, get_run_name, hcfg, hmain
from pycarus.geometry.pcd import random_point_sampling
from pycarus.utils import progress_bar
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from task_mapping_network.inr2vec.models.idecoder import ImplicitDecoder as INRDecoder
from task_mapping_network.inr2vec.models.transfer import Transfer
from models.idecoder import ImplicitDecoder as NeRFDecoder
from nerf.utils import Rays, render_image, render_image_GT
from nerf2vec.utils import generate_rays, pose_spherical

from nerfacc import OccupancyGrid
from nerf.intant_ngp import NGPradianceField
from nerf2vec import config as nerf2vec_config


logging.disable(logging.INFO)

class InrEmbeddingDataset(Dataset):
    def __init__(self, nerfs_root: Path, inrs_root: Path, split: str) -> None:
        super().__init__()

        self.nerfs_root = nerfs_root / split
        self.inrs_root = inrs_root / split

        self.nerf_item_paths = sorted(self.nerfs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.inr_item_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))

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


class CompletionTrainer:
    def __init__(self) -> None:

        inrs_dset_root = Path(hcfg("inrs_dset_root", str))  
        nerfs_dset_root = Path(hcfg("nerfs_dset_root", str))

        train_split = hcfg("train_split", str)
        train_dset = InrEmbeddingDataset(nerfs_dset_root, inrs_dset_root, train_split)

        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=8, shuffle=True)

        val_bs = hcfg("val_bs", int)
        val_split = hcfg("val_split", str)
        val_dset = InrEmbeddingDataset(nerfs_dset_root, inrs_dset_root, val_split)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=8, shuffle=True)
        self.train_val_loader = DataLoader(train_dset, batch_size=val_bs, num_workers=8, shuffle=True)

        embedding_dim = hcfg("embedding_dim", int)
        num_layers = hcfg("num_layers_transfer", int)
        transfer = Transfer(embedding_dim, num_layers)
        self.transfer = transfer.cuda()

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
        inr_decoder_ckpt_path = hcfg('inr2vec_decoder_ckpt_path', str)
        inr_decoder_ckpt = torch.load(inr_decoder_ckpt_path, map_location='cpu')
        inr_decoder.load_state_dict(inr_decoder_ckpt["decoder"])
        self.inr_decoder = inr_decoder.cuda()
        self.inr_decoder.eval()

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
        self.nerf_decoder = nerf_decoder.cuda()
        nerf2vec_decoder_ckpt_path = hcfg('nerf2vec_decoder_ckpt_path', str)
        print(f'loading nerf2vec weights: {nerf2vec_decoder_ckpt_path}')
        ckpt = torch.load(nerf2vec_decoder_ckpt_path, map_location='cpu')
        self.nerf_decoder.load_state_dict(ckpt["decoder"])
        
        # ####################
        # NerfAcc 
        # ####################
        self.device = settings.DEVICE_NAME

        occupancy_grid = OccupancyGrid(
            roi_aabb=nerf2vec_config.GRID_AABB,
            resolution=nerf2vec_config.GRID_RESOLUTION,
            contraction_type=nerf2vec_config.GRID_CONTRACTION_TYPE,
        )
        self.occupancy_grid = occupancy_grid.to(self.device)
        self.occupancy_grid.eval()

        self.ngp_mlp = NGPradianceField(**nerf2vec_config.INSTANT_NGP_MLP_CONF).to(self.device)
        self.ngp_mlp.eval()

        self.scene_aabb = torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=self.device)
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            * math.sqrt(3)
            / nerf2vec_config.GRID_CONFIG_N_SAMPLES
        ).item()

        lr = hcfg("lr", float)
        wd = hcfg("wd", float)
        self.optimizer = AdamW(self.transfer.parameters(), lr, weight_decay=wd)

        self.epoch = 0
        self.global_step = 0
        self.best_chamfer = 1000.0

        base_out_dir = Path(hcfg("out_root", str))
        self.ckpts_path = base_out_dir / "ckpts"
        self.all_ckpts_path = base_out_dir / "all_ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True, parents=True)
        self.all_ckpts_path.mkdir(exist_ok=True, parents=True)

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def train(self) -> None:
        num_epochs = hcfg("num_epochs", int)
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            self.transfer.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                embedding_nerf, _, _, embedding_pcd, _ = batch
                embedding_nerf = embedding_nerf.cuda()
                embedding_pcd = embedding_pcd.cuda()

                embeddings_transfer = self.transfer(embedding_pcd)

                loss = F.mse_loss(embeddings_transfer, embedding_nerf)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})

                self.global_step += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                # Validation and plotting have been commented because not used in the experimentations

                # self.val("train")
                # self.val("val")
                # self.plot("train")
                # self.plot("val")
                self.save_ckpt()
            
            """
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                self.plot("train")
                self.plot("val")
            """

            if epoch % 50 == 0:
                self.save_ckpt(all=True)
    """
    def val(self, split: str) -> None:
        loader = self.train_val_loader if split == "train" else self.val_loader
        self.transfer.eval()

        cdts = []
        fscores = []
        idx = 0

        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            incompletes, completes, embeddings_incomplete, embeddings_complete = batch
            bs = incompletes.shape[0]
            incompletes = incompletes.cuda()
            completes = completes.cuda()
            embeddings_incomplete = embeddings_incomplete.cuda()
            embeddings_complete = embeddings_complete.cuda()

            with torch.no_grad():
                embeddings_transfer = self.transfer(embeddings_incomplete)

            def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
                emb = embeddings_transfer[indices]
                pred = torch.sigmoid(self.decoder(emb, coords))
                pred = 1 - pred
                pred *= 0.1
                return pred

            pred_pcds = sample_pcds_from_udfs(udfs_func, bs, 4096, (-1, 1), 0.05, 0.02, 2048, 1)

            completes_2048 = random_point_sampling(completes, 2048)

            cd = chamfer_t(pred_pcds, completes_2048)
            cdts.extend([float(cd[i]) for i in range(bs)])

            f = f_score(pred_pcds, completes_2048, threshold=0.01)[0]
            fscores.extend([float(f[i]) for i in range(bs)])

            if idx > 9:
                break
            idx += 1

        mean_cdt = sum(cdts) / len(cdts)
        mean_fscore = sum(fscores) / len(fscores)

        self.logfn({f"{split}/cdt": mean_cdt})
        self.logfn({f"{split}/fscore": mean_fscore})

        if split == "val" and mean_cdt < self.best_chamfer:
            self.best_chamfer = mean_cdt
            self.save_ckpt(best=True)
    """
    
    @torch.no_grad()
    def plot(self, split: str) -> None:
        loader = self.train_val_loader if split == "train" else self.val_loader
        self.transfer.eval()

        loader_iter = iter(loader)
        batch = next(loader_iter)

        embedding_nerfs, nerf_data_dirs, pcds, embedding_pcds, uuid_pcds = batch

        grids = []
        nerfs = []
        for nerf_data_dir in nerf_data_dirs:
            nerf_path = os.path.join(nerf_data_dir, nerf2vec_config.NERF_WEIGHTS_FILE_NAME)
            nerf = torch.load(nerf_path, map_location=torch.device('cpu'))  
            nerfs.append(nerf)

            grid_weights_path = os.path.join(nerf_data_dir, 'grid.pth')  
            grid = torch.load(grid_weights_path, map_location='cpu')
            grid['_binary'] = grid['_binary'].to_dense()
            n_total_cells = nerf2vec_config.GRID_NUMBER_OF_CELLS
            grid['occs'] = torch.empty([n_total_cells]) 
            grids.append(grid)

        bs = pcds.shape[0]
        pcds = pcds.cuda()
        # nerfs = nerfs.cuda()
        embedding_pcds = embedding_pcds.cuda()
        embedding_nerfs = embedding_nerfs.cuda()

        with torch.no_grad():
            embeddings_transfer = self.transfer(embedding_pcds)

        pcds_2048 = random_point_sampling(pcds, 2048)

        # NeRF rendering parameters
        width = 224
        height = 224
        camera_angle_x = 0.8575560450553894 # Parameter taken from trained NeRFs
        focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

        max_images = 1
        array = np.linspace(-30.0, 30.0, max_images//2, endpoint=False)
        array = np.append(array, np.linspace(
            30.0, -30.0, max_images//2, endpoint=False))
    
    
        theta = 90.0  # [0, 360]
        gamma = -30  # [30.0, -30].
        distance = 1.5
        c2w = pose_spherical(torch.tensor(theta), torch.tensor(gamma), torch.tensor(distance))
        c2w = c2w.cuda()
        rays = generate_rays(embeddings_transfer.device, width, height, focal_length, c2w)

        color_bkgds = torch.ones((1,3)) 
        color_bkgds = color_bkgds.cuda()

        for idx in range(bs):

            print(f'Processing element {idx+1}/{bs}')
            
            rgb_pred, _, _, _, _, _ = render_image(
                radiance_field=self.nerf_decoder,
                embeddings=embeddings_transfer[idx].unsqueeze(dim=0),
                occupancy_grid=None,
                rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                scene_aabb=self.scene_aabb,
                render_step_size=self.render_step_size,
                render_bkgd=color_bkgds.unsqueeze(dim=0),
                grid_weights=None,
                device=self.device
            )
            
            print('render_image')
            
            curr_grid_weights = {
                '_roi_aabb': [grids[idx]['_roi_aabb']],
                '_binary': [grids[idx]['_binary']],
                'resolution': [grids[idx]['resolution']],
                'occs': [grids[idx]['occs']],
            }
            
            
            nerfs[idx]['mlp_base.params'] = [nerfs[idx]['mlp_base.params']]

            rgb_gt, _, _ = render_image_GT(
                radiance_field=self.ngp_mlp, 
                occupancy_grid=self.occupancy_grid, 
                rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)), 
                scene_aabb=self.scene_aabb, 
                render_step_size=self.render_step_size,
                color_bkgds=color_bkgds.unsqueeze(dim=0),
                grid_weights=curr_grid_weights,
                ngp_mlp_weights=nerfs[idx],
                device=self.device,
                training=False
            )
            print('render_image_GT')
                            
            pcd_wo3d = wandb.Object3D(pcds_2048[idx].cpu().detach().numpy())
            nerf_pred = wandb.Image((rgb_pred.to('cpu').detach().numpy() * 255).astype(np.uint8))
            nerf_gt = wandb.Image((rgb_gt.to('cpu').detach().numpy() * 255).astype(np.uint8))

            nerf_logs = {f"{split}/nerf_{idx}": [nerf_gt, nerf_pred]}
            pcd_logs = {f"{split}/pcd_{idx}": [pcd_wo3d]}
            self.logfn(nerf_logs)
            self.logfn(pcd_logs)

    def save_ckpt(self, best: bool = False, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_chamfer": self.best_chamfer,
            "net": self.transfer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_chamfer = ckpt["best_chamfer"]

            self.transfer.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])


@hmain(
    base_cfg_dir="task_mapping_network/cfg/bases",
    template_cfg_file="task_mapping_network/cfg/completion.yaml",
    run_cfg_file=None,
    parse_cmd_line=False,
)
def main() -> None:
    wandb.init(
        entity="entity",
        project="mapping_network",
        name=get_run_name(),
        dir=str(hcfg("out_root", str)),
        config=get_cfg_copy(),
    )

    trainer = CompletionTrainer()
    trainer.train()


if __name__ == "__main__":
    main()