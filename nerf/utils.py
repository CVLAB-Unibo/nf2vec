import math
import random
import time
from typing import Optional

import numpy as np
import torch
import collections
import torch.nn.functional as F

from nerfacc import OccupancyGrid, contract_inv, ray_marching, render_visibility, rendering

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def render_image(
    # scene
    radiance_field: torch.nn.Module,             
    embeddings: torch.Tensor,
    occupancy_grid: OccupancyGrid,               
    rays: Rays,                                  
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,  
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # other configs
    grid_weights:dict = None,
    background_indices = None,
    max_foreground_coordinates = 25000,
    max_background_coordinates = 10000,
    device='cuda:0'
):
    """Render the pixels of an image."""

    rays_shape = rays.origins.shape
    if len(rays_shape) == 4:
        batch_size, height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([batch_size] + [num_rays] + list(r.shape[3:])), rays
        )
    else:
        batch_size, num_rays, _ = rays_shape
 
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        _ = t_starts
        _ = t_ends
        _ = ray_indices

        rgb =  curr_rgb[curr_batch_idx][curr_mask] if curr_mask is not None else curr_rgb[curr_batch_idx]
        sigmas = curr_sigmas[curr_batch_idx][curr_mask] if curr_mask is not None else curr_sigmas[curr_batch_idx]

        return rgb, sigmas

    results = []
    
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training or batch_size > 1
        else 1024
    )
    
    MAX_SIZE = max_foreground_coordinates

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[:, i : i + chunk], rays)

        b_positions = []
        b_t_starts = []
        b_t_ends = []
        b_ray_indices = []
        max_bg_padding = 0

        # ####################
        # RAY MARCHING
        # ####################
        with torch.no_grad():
            
            for batch_idx in range(batch_size):

                if grid_weights != None and occupancy_grid != None:
                    dict = {
                        '_roi_aabb': grid_weights['_roi_aabb'][batch_idx],
                        '_binary': grid_weights['_binary'][batch_idx],
                        'resolution': grid_weights['resolution'][batch_idx],
                        'occs': grid_weights['occs'][batch_idx],
                    }
                    # occupancy_grid = None
                    occupancy_grid.load_state_dict(dict)

                ray_indices, t_starts, t_ends = ray_marching(
                    chunk_rays.origins[batch_idx],  
                    chunk_rays.viewdirs[batch_idx], 
                    scene_aabb=scene_aabb, 
                    grid=occupancy_grid,
                    sigma_fn=None,  # Different from NerfAcc. Not beneficial/useful for nerf2vec.
                    near_plane=near_plane,
                    far_plane=far_plane,
                    render_step_size=render_step_size,
                    stratified=radiance_field.training,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )

                b_t_starts.append(t_starts)
                b_t_ends.append(t_ends)
                b_ray_indices.append(ray_indices)
                
                # Compute positions
                t_origins = chunk_rays.origins[batch_idx][ray_indices]
                t_dirs = chunk_rays.viewdirs[batch_idx][ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
                b_positions.append(positions)

            padding_masks = [None]*batch_size

            if radiance_field.training or batch_size > 1:

                for batch_idx in range(batch_size):
                    
                    if background_indices is not None:
                        grid_coords = occupancy_grid.grid_coords[background_indices][batch_idx]
                        bg_positions = (
                            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
                        ) / occupancy_grid.resolution

                        bg_positions = contract_inv(
                            bg_positions,
                            roi=occupancy_grid._roi_aabb,
                            type=occupancy_grid._contraction_type,
                        )

                    # PADDING: do nothing, the padding will be added later
                    if b_positions[batch_idx].size(0) < MAX_SIZE:
                        pass

                    # TRUNCATION: Randomly sample MAX_SIZE elements from the tensor
                    else:
                        n_elements = b_positions[batch_idx].shape[0]
                        indices = torch.randperm(n_elements, device=device)[:MAX_SIZE]
                        indices, _ = torch.sort(indices)  # This is important to avoid problem with volume rendering
                        b_positions[batch_idx] = b_positions[batch_idx][indices]
                        b_t_starts[batch_idx] = b_t_starts[batch_idx][indices]
                        b_t_ends[batch_idx] = b_t_ends[batch_idx][indices]
                        b_ray_indices[batch_idx] = b_ray_indices[batch_idx][indices]

                    
                    # Background management
                    if background_indices is not None:
                        # bg_positions
                        MAX_SIZE_WITH_BACKGRDOUND = MAX_SIZE + max_background_coordinates
                        initial_size = b_positions[batch_idx].size(0)
                        padding_size = MAX_SIZE_WITH_BACKGRDOUND - initial_size
                        
                        b_positions[batch_idx] = torch.cat([b_positions[batch_idx], bg_positions[:padding_size]], dim=0)
                        b_t_starts[batch_idx] = F.pad(b_t_starts[batch_idx], pad=(0, 0, 0, padding_size))
                        b_t_ends[batch_idx] = F.pad(b_t_ends[batch_idx], pad=(0, 0, 0, padding_size))
                        b_ray_indices[batch_idx] = F.pad(b_ray_indices[batch_idx], pad=(0, padding_size))
                        
                        # Create masks used for ignoring the padding
                        padding_masks[batch_idx] = torch.zeros(MAX_SIZE_WITH_BACKGRDOUND, dtype=torch.bool)
                        padding_masks[batch_idx][:initial_size] = True

                        # Keep track of the largest tensor that contains the highest number of background coords
                        if padding_size > max_bg_padding:
                            max_bg_padding = padding_size
                    
                                
            # Convert arrays in tensors        
            b_t_starts = torch.stack(b_t_starts, dim=0)
            b_t_ends = torch.stack(b_t_ends, dim=0)
            b_ray_indices = torch.stack(b_ray_indices, dim=0)
            b_positions = torch.stack(b_positions, dim=0)
            
        # ####################
        # VOLUME RENDERING
        # ####################
        curr_rgb, curr_sigmas = radiance_field(embeddings, b_positions)

        bg_rgb_pred = torch.empty((batch_size, max_bg_padding, 3), device=device)  
        bg_rgb_label = torch.empty((batch_size, max_bg_padding, 3), device=device) 

        for curr_batch_idx in range(batch_size):
            
            curr_mask = padding_masks[curr_batch_idx]
            rgb, opacity, depth = rendering(
                b_t_starts[curr_batch_idx][curr_mask] if curr_mask is not None else b_t_starts[curr_batch_idx],
                b_t_ends[curr_batch_idx][curr_mask] if curr_mask is not None else b_t_ends[curr_batch_idx],
                b_ray_indices[curr_batch_idx][curr_mask] if curr_mask is not None else b_ray_indices[curr_batch_idx],
                n_rays=chunk_rays.origins.shape[1], #original num_rays (important for the final output on which the loss will be computed)
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd[curr_batch_idx],
            )
            
            chunk_results = [rgb, opacity, depth, len(b_t_starts[curr_batch_idx])]
            
            # Append to results array
            if curr_batch_idx < len(results):
                results[curr_batch_idx].append(chunk_results)
            else:
                results.append([chunk_results])
        
            # Manage background as padding
            if curr_mask is not None:
                inverted_mask = ~curr_mask
                curr_bg_rgb_pred = curr_rgb[curr_batch_idx][inverted_mask]
                curr_bg_sigma_pred = curr_sigmas[curr_batch_idx][inverted_mask]
                
                # Background composition (refer to vol_rendering.py from NerfAcc)
                curr_bg_rgb_pred = curr_bg_rgb_pred * curr_bg_sigma_pred + render_bkgd[curr_batch_idx].clone() * (1.0 - curr_bg_sigma_pred)
                curr_bg_rgb_label = render_bkgd[curr_batch_idx].repeat(curr_bg_rgb_pred.shape[0], 1) 

                # Compute padding size
                initial_size = curr_bg_rgb_pred.shape[0]
                padding_size = max_bg_padding - initial_size
                
                # Add padding to rgb predictions
                curr_bg_rgb_pred = F.pad(curr_bg_rgb_pred, pad=(0, 0, 0, padding_size))
                bg_rgb_pred[curr_batch_idx] = curr_bg_rgb_pred

                # Add padding to labels
                curr_bg_rgb_label = F.pad(curr_bg_rgb_label, pad=(0, 0, 0, padding_size))
                bg_rgb_label[curr_batch_idx] = curr_bg_rgb_label

            
    colors, opacities, depths, n_rendering_samples = zip(*[
        (
            torch.cat([r[0] for r in batch], dim=0),
            torch.cat([r[1] for r in batch], dim=0),
            torch.cat([r[2] for r in batch], dim=0),
            [r[3] for r in batch]
        ) for batch in results
    ])
    
    
    colors = torch.stack(colors, dim=0).view((*rays_shape[:-1], -1))
    opacities = torch.stack(opacities, dim=0).view((*rays_shape[:-1], -1))
    depths = torch.stack(depths, dim=0).view((*rays_shape[:-1], -1))
    n_rendering_samples = [sum(tensor) for tensor in n_rendering_samples] 
    
    return (
        colors, opacities, depths, n_rendering_samples, bg_rgb_pred, bg_rgb_label
    )

@torch.no_grad()
def render_image_GT(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    color_bkgds: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    # other configs
    grid_weights  = None,
    ngp_mlp_weights= None,
    device='cuda:0',
    training=True,
    max_elements=512
):
    
    filtered_rays = rays

    rays_shape = rays.origins.shape
    is_rendering_full_image = True if len(rays_shape) == 4 else False
    
    if is_rendering_full_image:
        batch_size, height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([batch_size] + [num_rays] + list(r.shape[3:])), rays
        )
        pixels = torch.empty((batch_size, height, width, 3), device=device)
        b_opacities = torch.empty((batch_size, height, width, 1), device=device)

    else:
        batch_size, num_rays, _ = rays_shape
        
        pixels = torch.empty((batch_size, max_elements, 3), device=device)
        b_opacities = torch.empty((batch_size, max_elements, 1), device=device)
        rays_shape = torch.Size((batch_size, max_elements, rays_shape[-1]))
        filtered_rays = Rays(
            origins=torch.empty((batch_size, max_elements, 3), device=device), 
            viewdirs=torch.empty((batch_size, max_elements, 3), device=device)
        )
        

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        
        _, density = radiance_field._query_density_and_rgb(positions, None)
        return density

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)


    for batch_idx in range(batch_size):
        curr_bkgd = color_bkgds[batch_idx]
        curr_grid_weights = {
            '_roi_aabb': grid_weights['_roi_aabb'][batch_idx],
            '_binary': grid_weights['_binary'][batch_idx],
            'resolution': grid_weights['resolution'][batch_idx],
            'occs': grid_weights['occs'][batch_idx],
        }
        curr_ngp_mlp_weights = {
            'mlp_base.params': ngp_mlp_weights['mlp_base.params'][batch_idx]
        }

        radiance_field.load_state_dict(curr_ngp_mlp_weights)
        occupancy_grid.load_state_dict(curr_grid_weights)

        curr_rays = Rays(origins=rays.origins[batch_idx], viewdirs=rays.viewdirs[batch_idx])

        results = []
        
        chunk = (
            torch.iinfo(torch.int32).max
            if training
            else 4096
        )

        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], curr_rays)
            ray_indices, t_starts, t_ends = ray_marching(
                chunk_rays.origins,
                chunk_rays.viewdirs,
                scene_aabb=scene_aabb,
                grid=occupancy_grid,
                sigma_fn=sigma_fn,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                stratified=radiance_field.training,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            rgb, opacity, depth = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=curr_bkgd,
            )
            chunk_results = [rgb, opacity, depth, len(t_starts)]
            results.append(chunk_results)
        colors, opacities, _, _ = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        
        if not is_rendering_full_image:
            colors, opacities, indices = sample_pixels_uniformly(opacities, colors, max_elements)
    
            filtered_rays.origins[batch_idx] = rays.origins[batch_idx][indices]
            filtered_rays.viewdirs[batch_idx] = rays.viewdirs[batch_idx][indices]
        
        pixels[batch_idx] = colors.view((*rays_shape[1:-1], -1))
        b_opacities[batch_idx] = opacities.view((*rays_shape[1:-1], -1))
    
        
    return pixels, b_opacities, filtered_rays

def sample_pixels_uniformly(opacities, colors, max_elements):
    # Get indices of True and False elements, where True means that the specified coordinate 
    # contains the 3d model, False otherwise.

    true_indices =  torch.nonzero(opacities.squeeze()).squeeze()
    if len(true_indices) < max_elements:
        print('true_indices < max_elements')
        n_missing_elements = max_elements - len(true_indices)
        
        true_indices_pad = true_indices[-n_missing_elements:]
        true_indices = torch.cat((true_indices, true_indices_pad), dim=0)

    merged_indices = true_indices[:max_elements]
    new_colors = colors[merged_indices]
    new_opacities = opacities[merged_indices]

    return new_colors, new_opacities, merged_indices