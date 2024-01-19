from typing import Callable, Tuple, List, Union

import torch
from einops import repeat
from torch import Tensor, nn
import tinycudann as tcnn

from nerf.intant_ngp import _TruncExp

class CoordsEncoder:
    def __init__(
        self,
        encoding_conf: dict,
        input_dims: int = 3
    ) -> None:
        self.input_dims = input_dims

        self.coords_enc = tcnn.Encoding(input_dims, encoding_conf, seed=999)
        self.out_dim = self.coords_enc.n_output_dims

    def apply_encoding(self, x):
        return self.coords_enc(x)

    def embed(self, inputs: Tensor) -> Tensor:
        # return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        result_encoding = self.apply_encoding(inputs.view(-1, 3))
        result_encoding = result_encoding.view(inputs.size()[0],inputs.size()[1],-1)
        return result_encoding

class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers_before_skip: int,
        num_hidden_layers_after_skip: int,
        out_dim: int,
        encoding_conf: dict,  # Added for NerfAcc
        aabb: Union[torch.Tensor, List[float]]  # Added for NerfAcc
    ) -> None:
        super().__init__()

        self.coords_enc = CoordsEncoder(encoding_conf=encoding_conf, input_dims=in_dim)
        coords_dim = self.coords_enc.out_dim

        # ################################################################################
        # Added for NerfAcc
        # ################################################################################
        trunc_exp = _TruncExp.apply
        self.density_activation = lambda x: trunc_exp(x - 1)
        self.aabb = aabb
        self.in_dim = in_dim
        # ################################################################################

        self.in_layer = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        self.skip_proj = nn.Sequential(nn.Linear(embed_dim + coords_dim, hidden_dim), nn.ReLU())

        before_skip = []
        for _ in range(num_hidden_layers_before_skip):
            before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(num_hidden_layers_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        self.after_skip = nn.Sequential(*after_skip)

    def forward(self, embeddings: Tensor, coords: Tensor) -> Tensor:
        

        # Sometimes the ray march algorithm calls the model with an input with 0 length.
        # The CutlassMLP crashes in these cases, therefore this fix has been applied.
        batch_size, n_coords, _ = coords.size()
        if n_coords == 0:
            rgb = torch.zeros([batch_size, 0, 3], device=coords.device)
            density = torch.zeros([batch_size, 0, 1], device=coords.device)
            return rgb, density

        # ################################################################################
        # Added for NerfAcc
        # ################################################################################
        aabb_min, aabb_max = torch.split(self.aabb, self.in_dim, dim=-1)
        coords = (coords - aabb_min) / (aabb_max - aabb_min)
        selector = ((coords > 0.0) & (coords < 1.0)).all(dim=-1)
        # ################################################################################

        coords = self.coords_enc.embed(coords)

        repeated_embeddings = repeat(embeddings, "b d -> b n d", n=coords.shape[1])

        emb_and_coords = torch.cat([repeated_embeddings, coords], dim=-1)

        x = self.in_layer(emb_and_coords)
        x = self.before_skip(x)

        inp_proj = self.skip_proj(emb_and_coords)
        x = x + inp_proj

        x = self.after_skip(x)
        # return x.squeeze(-1) # ORIGINAL INR2VEC IMPLEMENTATION

        # ################################################################################
        # Added for NerfAcc
        # ################################################################################
        rgb, density_before_activation = x[..., :3], x[..., 3]
        density_before_activation = density_before_activation[:, :, None]

        # Be sure that the density is non-negative
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )

        rgb = torch.nn.Sigmoid()(rgb)

        return rgb, density
        # ################################################################################

