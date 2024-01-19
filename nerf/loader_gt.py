"""
The NeRFLoderGT class inherits from Dataset, but it's not used as a Dataset in the training loop. This because the current
implementation was inherited from the original NerfAcc implementation. In the future, it could be useful to remove this dependency.
"""
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from nerf.utils import Rays

def _load_renderings(data_dir: str, split: str, h: int, w: int):
    
    with open(
        os.path.join(data_dir, "transforms_{}_compressed.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)

    camtoworlds = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        # fname = os.path.join(data_dir, frame["file_path"] + ".png")
        camtoworlds.append(frame["transform_matrix"])

    camtoworlds = np.stack(camtoworlds, axis=0)

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return  camtoworlds, focal


class NeRFLoaderGT(torch.utils.data.Dataset):

    WIDTH, HEIGHT = 224, 224  
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        color_bkgd_aug: str = "random",  
        num_rays: int = None,
        near: float = None,
        far: float = None,
        device: str = "cuda:0",
        weights_file_name: str = "nerf_weights.pth",
        training: bool = True
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far

        self.training = training

        self.device = device

        self.color_bkgd_aug = color_bkgd_aug

        self.weights_file_path = os.path.join(data_dir, weights_file_name)
        
        self.camtoworlds, self.focal = _load_renderings(#_from_RAM(
            data_dir, split, self.HEIGHT, self.WIDTH
        )
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(self.device).to(torch.float32)
        )
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )  # (3, 3)

    def __len__(self):
        return len(self.camtoworlds)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rays = data["rays"]
        # pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.device)
        else:
            color_bkgd = torch.zeros(3, device=self.device)

        # pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""

        num_rays = self.num_rays

        if self.training:
            camtoworld_id = torch.randint(
                0,
                len(self.camtoworlds),
                size=(num_rays,),
                device=self.device,
            )

            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.device
            )
        else:
            camtoworld_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.device),
                torch.arange(self.HEIGHT, device=self.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        c2w = self.camtoworlds[camtoworld_id]  # (num_rays, 3, 4)

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        
        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rays": rays,  # [h, w, 3] 
        }
