import os
from typing import List, Optional
import logging

import hydra
from hydra.utils import  instantiate
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

log = logging.getLogger(__name__)


def train(config: DictConfig) -> Optional[float]:
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.datamodule)

    # # Init lightning model
    # log.info(f"Instantiating model <{config.model._target_}>")
    # model: LightningModule = instantiate(config.model)
    #
    # # Init lightning callbacks
    # callbacks: List[Callback] = []
    # if "callbacks" in config:
    #     for _, cb_conf in config.callbacks.items():
    #         if "_target_" in cb_conf:
    #             log.info(f"Instantiating callback <{cb_conf._target_}>")
    #             callbacks.append(instantiate(cb_conf))
    #
    # # Init lightning loggers
    # logger: List[LightningLoggerBase] = []
    # if "logger" in config:
    #     for _, lg_conf in config.logger.items():
    #         if "_target_" in lg_conf:
    #             log.info(f"Instantiating logger <{lg_conf._target_}>")
    #             logger.append(instantiate(lg_conf))
    #
    # # Init lightning trainer
    # log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # trainer: Trainer = instantiate(
    #     config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    # )
    #
    # # Send some parameters from config to all lightning loggers
    # log.info("Logging hyperparameters!")
    # utils.log_hyperparameters(
    #     config=config,
    #     model=model,
    #     datamodule=datamodule,
    #     trainer=trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    # )
    #
    # # Train the model
    # if config.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(model=model, datamodule=datamodule)
    #
    # # Get metric score for hyperparameter optimization
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric and optimized_metric not in trainer.callback_metrics:
    #     raise Exception(
    #         "Metric for hyperparameter optimization not found! "
    #         "Make sure the `optimized_metric` in `hparams_search` config is correct!"
    #     )
    # score = trainer.callback_metrics.get(optimized_metric)
    #
    # # Test the model
    # if config.get("test"):
    #     ckpt_path = "best"
    #     if not config.get("train") or config.trainer.get("fast_dev_run"):
    #         ckpt_path = None
    #     log.info("Starting testing!")
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #
    # # Make sure everything closed properly
    # log.info("Finalizing!")
    # utils.finish(
    #     config=config,
    #     model=model,
    #     datamodule=datamodule,
    #     trainer=trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    # )
    #
    # # Print path to best checkpoint
    # if not config.trainer.get("fast_dev_run") and config.get("train"):
    #     log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score


# def test(config: DictConfig) -> None:
#     """Contains minimal example of the testing pipeline.
#     Evaluates given checkpoint on a testset.
#
#     Args:
#         config (DictConfig): Configuration composed by Hydra.
#
#     Returns:
#         None
#     """
#
#     # Set seed for random number generators in pytorch, numpy and python.random
#     if config.get("seed"):
#         seed_everything(config.seed, workers=True)
#
#     # Convert relative ckpt path to absolute path if necessary
#     if not os.path.isabs(config.ckpt_path):
#         config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)
#
#     # Init lightning datamodule
#     log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
#     datamodule: LightningDataModule = instantiate(config.datamodule)
#
#     # Init lightning model
#     log.info(f"Instantiating model <{config.model._target_}>")
#     model: LightningModule = instantiate(config.model)
#
#     # Init lightning loggers
#     logger: List[LightningLoggerBase] = []
#     if "logger" in config:
#         for _, lg_conf in config.logger.items():
#             if "_target_" in lg_conf:
#                 log.info(f"Instantiating logger <{lg_conf._target_}>")
#                 logger.append(instantiate(lg_conf))
#
#     # Init lightning trainer
#     log.info(f"Instantiating trainer <{config.trainer._target_}>")
#     trainer: Trainer = instantiate(config.trainer, logger=logger)
#
#     # Log hyperparameters
#     trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})
#
#     log.info("Starting testing!")
#     trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
