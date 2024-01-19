import json
import os
from anyio import Path
from hesiod import hcfg, hmain
import open3d as o3d

from typing import List


def get_dataset_json(root:str, split: str):
        json_path = os.path.join(root, f'{split}.json')
        
        folders = []
        
        with open(json_path) as file:
            dset = json.load(file)
        
        for nerf_path in dset:
            # Skip augmented data
            if nerf_path.endswith('_A1') or nerf_path.endswith('_A2'):
                continue

            full_path = Path(nerf_path)
            relative_path = os.path.join(full_path.parts[-2], full_path.parts[-1])
            folders.append(relative_path)
        
        return folders

@hmain(
    base_cfg_dir="task_mapping_network/cfg/bases",
    template_cfg_file="task_mapping_network/cfg/pcd_dataset.yaml",
    run_cfg_file=None,
    parse_cmd_line=False,
    out_dir_root="task_mapping_network/logs"
)
def create_dataset(): 

    split_json_root_path = hcfg("split_json_root_path", str)
    out_point_clouds_path = hcfg("out_point_clouds_path", str)
    mesh_root = hcfg("shapenet_root", str)


    splits = hcfg("splits", List[str])
    

    for split in splits:
        shapes = get_dataset_json(split_json_root_path, split)
        for shape in shapes:
            
            mesh_class = shape.split('/')[0]
            mesh_id = shape.split('/')[1]

            mesh_path = os.path.join(mesh_root, shape, 'model.obj')
            

            mesh = o3d.io.read_triangle_mesh(mesh_path)

            num_points = 10000  # Adjust the number of points as needed
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            pcd_folder = os.path.join(out_point_clouds_path, mesh_class, split)
            os.makedirs(pcd_folder, exist_ok=True)
            
            pcd_full_path = os.path.join(pcd_folder, f'{mesh_id}.ply')

            o3d.io.write_point_cloud(pcd_full_path, pcd)
        

if __name__ == "__main__":
    create_dataset()
        

        

