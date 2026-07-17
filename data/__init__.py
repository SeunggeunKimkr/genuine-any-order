# -------------------------------------------------------
# Dataset bundle code
# -------------------------------------------------------

from omegaconf import DictConfig
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .tiny_gsm import split_tinygsm
from .tiny_gsm_split_v2 import split_tinygsm as split_tinygsm_split_v2


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    tokenizer: Optional[AutoTokenizer] = None


def setup_data_bundle(config: DictConfig) -> DatasetBundle:
    """
    get the dataset config and return the dataset bundle
    """
    tokenizer = None

    if config.dataset == "tinygsm":
        train_data, val_data = split_tinygsm(config.data_dir, val_ratio=config.val_ratio, seed=config.seed)
    elif config.dataset == "tinygsm_split_v2":
        train_data, val_data = split_tinygsm_split_v2(config.data_dir, val_ratio=config.val_ratio, seed=config.seed)
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")
    
    train_loader = DataLoader(
        train_data,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.training.cpus,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.training.cpus,
    )
    return DatasetBundle(train_loader, val_loader, tokenizer)


if __name__ == "__main__":
    # tinygsm test code
    config = DictConfig({
        "dataset": "tinygsm_split_v2",
        "data_dir": "data/tiny_gsm_split_v2",
        "val_ratio": 0.05,
        "seed": 2025,
        "training": {
            "per_gpu_batch_size": 16,
            "cpus": 16,
        }
    })
    bundle = setup_data_bundle(config)
    print(bundle)
