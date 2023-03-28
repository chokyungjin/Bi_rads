import os
from typing import Any, List, Optional

import hydra
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer, seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase
from src import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init pytorch model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(
        config.model, num_classes=datamodule.data_train.num_classes
    )

    # # Init loss
    # log.info(f"Instantiating loss <{config.loss._target_}>")
    # loss: Any = hydra.utils.instantiate(config.loss)

    # Init optimizer
    log.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    optimizer: Any = hydra.utils.instantiate(
        config.optimizer, params=model.parameters()
    )

    # Init scheduler
    log.info(f"Instantiating scheduler <{config.scheduler._target_}>")
    scheduler: Any = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)

    # Init lightning model
    log.info(f"Instantiating lightning engine <{config.engine._target_}>")
    engine: LightningModule = hydra.utils.instantiate(
        config.engine,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        _recursive_=False,
    )

    """Init lightning callbacks"""
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    """Init lightning loggers"""
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    """Init lightning trainer"""
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        # strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=engine,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=engine, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        trainer.test(model=engine, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=engine,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
