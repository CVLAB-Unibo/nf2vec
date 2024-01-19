import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import copy
import h5py
import datetime
import settings

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset


from models.fc_classifier import FcClassifier
from torchmetrics.classification.accuracy import Accuracy

from task_classification import config as classification_config
from nerf2vec import config as nerf2vec_config

import wandb

class InrEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            
        return embedding, class_id
    
class InrEmbeddingClassifier:
    def __init__(self, device='cuda:0') -> None:

        dset_root = Path(settings.NERF2VEC_EMBEDDINGS_DIR)
        train_dset = InrEmbeddingDataset(dset_root, nerf2vec_config.TRAIN_SPLIT)

        train_bs = classification_config.TRAIN_BS
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=8, shuffle=True)

        val_bs = classification_config.VAL_BS
        val_dset = InrEmbeddingDataset(dset_root, nerf2vec_config.VAL_SPLIT)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=8)

        test_dset = InrEmbeddingDataset(dset_root, nerf2vec_config.TEST_SPLIT)
        self.test_loader = DataLoader(test_dset, batch_size=val_bs, num_workers=8)

        layers_dim = classification_config.LAYERS_DIM
        self.num_classes = nerf2vec_config.NUM_CLASSES
        net = FcClassifier(layers_dim, self.num_classes)
        self.net = net.to(device)

        self.optimizer = AdamW(self.net.parameters(), classification_config.LR, weight_decay=classification_config.WD)
        num_steps = classification_config.NUM_EPOCHS * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, classification_config.LR, total_steps=num_steps)

        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

        self.ckpts_path = Path(settings.CLASSIFICATION_OUTPUT_DIR) / "ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(parents=True, exist_ok=True)
    
    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def interpolate(self, params: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        new_params = []
        new_labels = []

        for i, p1 in enumerate(params):
            same_class = labels == labels[i]
            num_same_class = torch.sum(same_class)

            if num_same_class > 2:
                indices = torch.where(same_class)[0]
                random_order = torch.randperm(len(indices))
                random_idx = indices[random_order][0]
                p2 = params[random_idx]

                random_uniform = torch.rand(len(p2))
                tsh = torch.rand(())
                from2 = random_uniform >= tsh
                p1_copy = p1.clone()
                p1_copy[from2] = p2[from2]
                new_params.append(p1_copy)
                new_labels.append(labels[i])

        if len(new_params) > 0:
            new_params = torch.stack(new_params)
            final_params = torch.cat([params, new_params], dim=0)
            new_labels = torch.stack(new_labels)
            final_labels = torch.cat([labels, new_labels], dim=0)
        else:
            final_params = params
            final_labels = labels

        return final_params, final_labels

    def train(self) -> None:
        
        self.config_wandb()

        num_epochs = classification_config.NUM_EPOCHS
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            print(f'epoch {epoch}/{num_epochs} started...')
            self.epoch = epoch

            self.net.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in self.train_loader:
                embeddings, labels = batch
                embeddings = embeddings.cuda()
                labels = labels.cuda()

                embeddings, labels = self.interpolate(embeddings, labels)

                pred = self.net(embeddings)
                loss = F.cross_entropy(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})

                self.global_step += 1

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.val("train")
                self.val("val")
                self.save_ckpt()

            if epoch == num_epochs - 1:
                predictions, true_labels = self.val("test", best=True)
                self.log_confusion_matrix(predictions, true_labels)

    @torch.no_grad()
    def val(self, split: str, best: bool = False) -> Tuple[Tensor, Tensor]:
        acc = Accuracy("multiclass", num_classes=self.num_classes).cuda()
        predictions = []
        true_labels = []

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        if best:
            model = self.best_model
        else:
            model = self.net
        model = model.cuda()
        model.eval()

        losses = []
        for batch in loader:
            params, labels = batch
            params = params.cuda()
            labels = labels.cuda()

            pred = self.net(params)
            loss = F.cross_entropy(pred, labels)
            losses.append(loss.item())

            pred_softmax = F.softmax(pred, dim=-1)
            acc(pred_softmax, labels)

            predictions.append(pred_softmax.clone())
            true_labels.append(labels.clone())

        accuracy = acc.compute()

        self.logfn({f"{split}/acc": accuracy})
        self.logfn({f"{split}/loss": torch.mean(torch.tensor(losses))})

        if accuracy > self.best_acc and split == "val":
            self.best_acc = accuracy
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.net)

        return torch.cat(predictions, dim=0), torch.cat(true_labels, dim=0)

    def log_confusion_matrix(self, predictions: Tensor, labels: Tensor) -> None:
        conf_matrix = wandb.plot.confusion_matrix(
            probs=predictions.cpu().numpy(),
            y_true=labels.cpu().numpy(),
            class_names=[str(i) for i in range(nerf2vec_config.NUM_CLASSES)],
        )
        self.logfn({"conf_matrix": conf_matrix})

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_acc": self.best_acc,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

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
            self.best_acc = ckpt["best_acc"]

            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
    
    def config_wandb(self):
        wandb.init(
            entity='dsr-lab',
            project='nerf2vec_classifier',
            name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=classification_config.WANDB_CONFIG
        )

def main() -> None:
    classifier = InrEmbeddingClassifier(device=settings.DEVICE_NAME)
    classifier.train()

if __name__ == "__main__":
    main()