import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import json
import h5py
import torch
import settings

from pathlib import Path
from typing import Tuple
from torch import Tensor
from models.encoder import Encoder
from nerf2vec import config as nerf2vec_config
from torch.utils.data import DataLoader, Dataset
from nerf2vec.utils import get_class_label, get_mlp_params_as_matrix

class InrDataset(Dataset):
    def __init__(self, split_json: str, device: str, nerf_weights_file_name: str) -> None:
        super().__init__()

        with open(split_json) as file:
            self.nerf_paths = json.load(file)
        
        # self.nerf_paths = self._get_nerf_paths('data\\data_TRAINED')
        assert isinstance(self.nerf_paths, list), 'The json file provided is not a list.'

        self.device = device
        self.nerf_weights_file_name = nerf_weights_file_name

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:

        data_dir = self.nerf_paths[index]
        weights_file_path = os.path.join(data_dir, self.nerf_weights_file_name)

        class_label = get_class_label(weights_file_path)
        class_id = nerf2vec_config.LABELS_TO_IDS[get_class_label(weights_file_path)] if class_label != -1 else class_label

        matrix = torch.load(weights_file_path, map_location=torch.device(self.device))
        matrix = get_mlp_params_as_matrix(matrix['mlp_base.params'])

        return matrix, class_id, data_dir

def load_nerf2vec_checkpoint():
    ckpts_path = Path(settings.NERF2VEC_CKPTS_PATH)
    ckpt_paths = [p for p in ckpts_path.glob("*.pt") if "best" not in p.name]
    error_msg = "Expected only one ckpt apart from best, found none or too many."
    assert len(ckpt_paths) == 1, error_msg
    ckpt_path = ckpt_paths[0]
    print(f'loading path: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    
    return ckpt


def export_embeddings(device = 'cuda:0'):

    train_dset_json = os.path.abspath(os.path.join('data', 'train.json'))
    train_dset = InrDataset(train_dset_json, device='cpu', nerf_weights_file_name=nerf2vec_config.NERF_WEIGHTS_FILE_NAME)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset_json = os.path.abspath(os.path.join('data', 'validation.json'))
    val_dset = InrDataset(val_dset_json, device='cpu', nerf_weights_file_name=nerf2vec_config.NERF_WEIGHTS_FILE_NAME)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset_json = os.path.abspath(os.path.join('data', 'test.json'))
    test_dset = InrDataset(test_dset_json, device='cpu', nerf_weights_file_name=nerf2vec_config.NERF_WEIGHTS_FILE_NAME)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    encoder = Encoder(
            nerf2vec_config.MLP_UNITS,
            nerf2vec_config.ENCODER_HIDDEN_DIM,
            nerf2vec_config.ENCODER_EMBEDDING_DIM
            )
    encoder = encoder.to(device)
    ckpt = load_nerf2vec_checkpoint()
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    
    loaders = [train_loader, val_loader, test_loader]
    splits = [nerf2vec_config.TRAIN_SPLIT, nerf2vec_config.VAL_SPLIT, nerf2vec_config.TEST_SPLIT]


    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in loader:
            matrices, class_ids, data_dirs = batch
            matrices = matrices.cuda()

            with torch.no_grad():
                embeddings = encoder(matrices)

            out_root = Path(settings.NERF2VEC_EMBEDDINGS_DIR)
            h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "w") as f:
                # print(f'dir: {data_dirs[0]}, class: {class_ids[0]}')
                f.create_dataset("data_dir", data=data_dirs[0])
                f.create_dataset("embedding", data=embeddings[0].detach().cpu().numpy())
                f.create_dataset("class_id", data=class_ids[0].detach().cpu().numpy())

            idx += 1

            if idx % 5000 == 0:
                print(f'Created {idx} embeddings for {split} split')

def main() -> None:
    export_embeddings(device=settings.DEVICE_NAME)

if __name__ == "__main__":
    main()