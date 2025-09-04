import argparse
import os
from dataclasses import dataclass, fields
from typing import Any, List, Optional
import hydra
import lightning as L
import torch
import wandb
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import WandbLogger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
from tasks.autoencoder import AETask, AETaskConfig


@dataclass
class TrainConfig:
    task: AETaskConfig
    seed: int
    log_dir: str
    max_epochs: int
    accumulate_grad_batches: int 
    val_check_interval: int
    logger: dict
    sweep_id: Optional[str] = None
    model_checkpoint: Optional[dict] = None
    early_stopping: Optional[dict] = None
    gradient_clip_val: float = 1.0

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
OmegaConf.register_new_resolver("eval", eval)

def main(cfg: Optional[TrainConfig] = None, run_id: Optional[str] = None):
    L.seed_everything(cfg.seed)

    # If run_id is provided, use the associated config
    if run_id != None:
        assert cfg is None
        wandb_id = run_id
        api = wandb.Api()
        entity = "guillaume-lajoie"
        project = "hlm"
        run = api.run(f"{entity}/{project}/{run_id}")
        cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), run.config)

    # Add user to logger
    if "USER" in os.environ:
        cfg.logger.tags = [os.environ["USER"]]

    logger = hydra.utils.instantiate(cfg.logger)
    wandb_id = logger.experiment.path.split("/")[-1]
    
    # Setup checkpoint (with wandb ID as <dirpath>)
    callbacks = []
    if cfg.model_checkpoint:
        cfg.model_checkpoint.dirpath = os.path.join(
            cfg.log_dir, "checkpoints", wandb_id
        )
        callbacks.append(hydra.utils.instantiate(cfg.model_checkpoint))

    # Init config object
    cfg = OmegaConf.to_container(
        cfg=cfg,
        resolve=True,
        throw_on_missing=False,
        enum_to_str=False,
        structured_config_mode=SCMode.INSTANTIATE,
    )

    task = AETask(cfg.task)

     # Instantiate the trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator='gpu',
        enable_checkpointing=True if cfg.model_checkpoint else False,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
        gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=100,
        accumulate_grad_batches=cfg.accumulate_grad_batches,#
        precision=16
    )

    #trainer.validate(model=task)
    trainer.fit(
        model=task,
        ckpt_path=(
            os.path.join(cfg.model_checkpoint["dirpath"], "last.ckpt")
            if run_id != None
            else None
        ),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_id")
    args, _ = parser.parse_known_args()

    # If --wandb_id is provided, resume the associated run
    if args.wandb_id == None:
        hydra_wrapper = hydra.main(
            version_base=None, config_name="train", config_path="configs/"
        )
        hydra_wrapper(main)()
    else:
        main(cfg=None, run_id=args.wandb_id)