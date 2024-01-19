"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = False,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 3,
        encoding='Frequency',
        mlp='CutlassMLP',
        activation='Sine',
        n_hidden_layers=4,
        n_neurons=256,
        encoding_size=24
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32,)
        # NERF2VEC: Added persisten=False
        self.register_buffer("aabb", aabb, persistent=False)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded

        self.geo_feat_dim = geo_feat_dim if use_viewdirs else 0

        if self.use_viewdirs:
            single_mlp_encoding_config = {
                "otype": "Composite",
                "nested": [
                    # POSITION ENCODING
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Frequency",
                        "n_frequencies": 6,

                    },
                    # DIRECTION ENCODING
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 1,  # Determines the output's dimension, which is degree^2
                    },
                    # {"otype": "Identity", "n_bins": 4, "degree": 4},
                ]
            }
        else:
            if encoding == 'Frequency':
                single_mlp_encoding_config = {
                    "otype": "Frequency",
                    "n_frequencies": encoding_size
                }
            else:
                single_mlp_encoding_config = {
                    "otype": "Identity"
                }

        # print(f'*'*40)
        # print(f'Initializing model: \n- mlp: {mlp} - {n_hidden_layers} hidden layers - {n_neurons} neurons\n- activation: {activation.upper()}\n- encoding: {encoding.upper()} - size: {encoding_size}')
        # print(f'*'*40)
        self.mlp_base = tcnn.NetworkWithInputEncoding(
            seed=999,
            n_input_dims=self.num_dim+self.geo_feat_dim,
            n_output_dims=4,
            encoding_config=single_mlp_encoding_config,
            network_config={
                "otype": mlp,  # FullyFusedMLP, CutlassMLP
                "activation": activation,
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers
            },
        )

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            # This view actually seems to do nothing
            self.mlp_base(x.view(-1, self.num_dim))
            # change the shape of the tensor to [all dimension of x but last, 1 + the feature dimension]
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)  # Same dtype as x (the input)
        )

        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.view(-1, dir.shape[-1]))

            # Concatenation of the DENSITIY MLP and the encoded view direction
            h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.view(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        return rgb

    def _query_density_and_rgb(self, x, dir=None):

        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        if self.use_viewdirs:
            if dir is not None:
                dir = (dir + 1.0) / 2.0
                # d = self.direction_encoding(dir.view(-1, dir.shape[-1]))

                x = torch.cat([x, dir], dim=-1)
            else:
                # if self.random_tensor == None:
                # random = torch.ones(x.shape[0], self.geo_feat_dim, device=x.device).to(x)
                # all ones or zeros are detrimental for the loss. It is much better a random tensor.
                # random = self.random_tensor.repeat(x.shape[0], 1).to(device=x.device)
                random = torch.rand(
                    x.shape[0], self.geo_feat_dim, device=x.device)
                # random = torch.zeros(x.shape[0], self.geo_feat_dim, device=x.device).to(x)
                x = torch.cat([x, random], dim=-1)

        # Sometimes the ray march algorithm calls the model with an input with 0 length.
        # The CutlassMLP crashes in these cases, therefore this fix has been applied.
        if len(x) == 0:
            rgb = torch.zeros([0, 3], device=x.device)
            density = torch.zeros([0, 1], device=x.device)
            return rgb, density

        out = (
            # self.mlp_base(x.view(-1, self.num_dim))  # This view actually seems to do nothing
            # This view actually seems to do nothing
            self.mlp_base(x.view(-1, self.num_dim+self.geo_feat_dim))
            # change the shape of the tensor to [all dimension of x but last, 1 + the feature dimension]
            # .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)  # Same dtype as x (the input)
        )

        rgb, density_before_activation = out[..., :3], out[..., 3]
        density_before_activation = density_before_activation[:, None]

        # Be sure that the density is non-negative
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )

        rgb = torch.nn.Sigmoid()(rgb)

        return rgb, density

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        """
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

            # density, embedding = self.query_density(positions, return_feat=True)

            # rgb = self._query_rgb(directions, embedding=embedding)
        """

        rgb, density = self._query_density_and_rgb(positions, directions)

        # print(f'rgb.shape: {rgb.shape}')
        # print(f'density.shape: {density.shape}')

        return rgb, density
