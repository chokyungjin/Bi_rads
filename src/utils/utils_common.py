import os
import torch
import collections
from itertools import repeat
from typing import Any, List, Optional, Tuple, Union

import glob
import hydra
from .utils_file import read_yaml
from omegaconf import DictConfig


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        raise RuntimeError(f"{dirname} is already exist")


def read_config(filepath):
    """Reads a cfg file."""
    cfg_file = DictConfig(read_yaml(filepath))
    return cfg_file


def get_datamodule(cfg: DictConfig, verbose: bool = False):
    assert "datamodule" in cfg.keys()
    if verbose:
        print(cfg["datamodule"], end="\n\n")
    return hydra.utils.instantiate(cfg["datamodule"])


def get_model(cfg: DictConfig, num_classes: int = None, verbose: bool = False):
    assert "model" in cfg.keys()
    assert num_classes is not None
    if verbose:
        print(cfg["model"], end="\n\n")
    return hydra.utils.instantiate(cfg["model"], num_classes=num_classes)


def get_engine(cfg: DictConfig, datamodule, verbose: bool = False):
    assert "engine" in cfg.keys()
    # criterion = get_loss(cfg, verbose) # deprecated
    model = get_model(cfg, datamodule.data_train.num_classes, verbose)
    if verbose:
        print(cfg["engine"], end="\n\n")
    # return hydra.utils.instantiate(cfg["engine"], model=model, criterion=criterion) # deprecated
    return hydra.utils.instantiate(cfg["engine"], model=model)


def load_cfg(cfg: DictConfig, verbose: bool = False):
    """Loads instances with cfg"""
    datamodule = get_datamodule(cfg, verbose)
    engine = get_engine(cfg, datamodule, verbose)
    return datamodule, engine


def load_cfg_automatically(root: str, verbose: bool = False, load_weight: bool = True):
    """Read and Load cfg automatically"""
    cfg = read_config(os.path.join(root, ".hydra", "config.yaml"))
    datamodule, engine = load_cfg(cfg, verbose)
    if load_weight:
        weight_files = glob.glob(os.path.join(root, "checkpoints", "epoch*.ckpt"))
        assert len(weight_files) == 1
        weight = torch.load(weight_files[0], map_location="cpu")
        print(engine.load_state_dict(weight["state_dict"]), end="\n\n")
    return datamodule, engine


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
