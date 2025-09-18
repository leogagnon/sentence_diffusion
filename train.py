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
from tasks.diffusion import GaussianDiffusionTask, GaussianDiffusionTaskConfig
from tasks.finetune import FinetuneTask, FinetuneTaskConfig
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_info

os.environ["LATENT_CONTROL_CKPT_DIR"] = (
    "/network/scratch/l/leo.gagnon/sentence_diffusion/logs/checkpoints"
)
torch.set_float32_matmul_precision('medium')


@dataclass
class TaskConfig:
    ae: Optional[AETaskConfig] = None
    diffusion: Optional[GaussianDiffusionTaskConfig] = None
    finetune: Optional[FinetuneTaskConfig] = None


@dataclass
class TrainConfig:
    task: TaskConfig
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
    name: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
OmegaConf.register_new_resolver("eval", eval)


def main(cfg: Optional[TrainConfig] = None, run_id: Optional[str] = None):

    # If run_id is provided, use the associated config
    if run_id != None:
        assert cfg is None
        # Gather the run's config
        wandb_id = run_id
        api = wandb.Api()
        entity = "guillaume-lajoie"
        project = "sentence_diffusion"
        run = api.run(f"{entity}/{project}/{run_id}")
        cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), run.config)

        # Set the logger ID to the provided run_id (to resume the run)
        cfg.logger.id = wandb_id
    else:
        # Setup tags for wandb
        tags = [k for k in cfg.task.keys() if cfg.task[k] != None]
        # Add user to logger
        if "USER" in os.environ:
            tags += [os.environ["USER"]]
        cfg.logger.tags = tags

    logger = hydra.utils.instantiate(cfg.logger)
    wandb_id = logger.experiment.path.split("/")[-1]

    L.seed_everything(cfg.seed)

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

    if cfg.early_stopping is not None:
        rank_zero_info("Using early stopping!")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))

    # Init lightning module
    if cfg.task.diffusion != None:
        task = GaussianDiffusionTask(cfg.task.diffusion)
        cfg.task.diffusion = task.cfg

        # If the task is diffusion, add the autoencoder config to cfg
        run = wandb.Api().run(
            f"guillaume-lajoie/sentence_diffusion/{cfg.task.diffusion.pretrained_ae_id}"
        )
        cfg.task.ae = OmegaConf.merge(
            OmegaConf.structured(AETaskConfig), run.config["task"]["ae"]
        )
        
        if run.config["task"]["finetune"] is not None:
            cfg.task.finetune = OmegaConf.merge(
                OmegaConf.structured(FinetuneTaskConfig), run.config["task"]["finetune"]
            )
    elif cfg.task.ae != None:
        task = AETask(cfg.task.ae)
        cfg.task.ae = task.cfg

        run = wandb.Api().run(
            f"guillaume-lajoie/sentence_diffusion/{cfg.task.ae.pretrained_decoder_id}"
        )
        cfg.task.finetune = OmegaConf.merge(
            OmegaConf.structured(FinetuneTaskConfig), run.config["task"]["finetune"]
        )
    elif cfg.task.finetune != None:
        task = FinetuneTask(cfg.task.finetune)
        cfg.task.finetune = task.cfg
    else:
        raise ValueError("No task specified in config")

    # Give the whole TrainConfig to wandb
    if cfg.logger:
        logger.experiment.config.update(
            OmegaConf.to_container(OmegaConf.structured(cfg)), allow_val_change=True
        )
        
    # Instantiate the trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        enable_checkpointing=True if cfg.model_checkpoint else False,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval * cfg.accumulate_grad_batches,  # to account for accumulation
        gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=50,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        precision="16-mixed",
    )
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
            version_base=None, config_path="configs/", config_name="train"
        )
        hydra_wrapper(main)()
    else:
        main(cfg=None, run_id=args.wandb_id)
