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
from tasks.dlclm import DLCLMTask, DLCLMTaskConfig
from tasks.dcse import DCSETask, DCSETaskConfig
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True


@dataclass
class TaskConfig:
    ae: Optional[AETaskConfig] = None
    dlclm: Optional[DLCLMTaskConfig] = None
    dcse: Optional[DCSETaskConfig] = None


@dataclass
class TrainConfig:
    task: TaskConfig
    seed: int
    max_steps: int
    val_check_interval: int
    logger: dict
    strategy: str = "auto"
    log_dir: Optional[str] = None
    sweep_id: Optional[str] = None
    effective_batch_size: Optional[int] = None  # if None, no accumulation
    model_checkpoint: Optional[dict] = None
    early_stopping: Optional[dict] = None
    gradient_clip_val: Optional[float] = None
    name: Optional[str] = None
    precision: str = "bf16-mixed"
    limit_val_batches: Optional[int] = None
    compile: bool = False
    num_devices: Optional[int] = None


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
        project = "dlc_lm"
        run = api.run(f"{entity}/{project}/{run_id}")
        cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), run.config)

        # Set the logger ID to the provided run_id (to resume the run)
        cfg.logger.id = wandb_id
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        # Setup tags for wandb
        tags = [k for k in cfg.task.keys() if cfg.task[k] != None]
        # Add user to logger
        if "USER" in os.environ:
            tags += [os.environ["USER"]]
        cfg.logger.tags = tags
        logger = hydra.utils.instantiate(cfg.logger)
        if rank_zero_only.rank == 0:
            wandb_id = logger.experiment.path.split("/")[-1]
        else:
            wandb_id = "dummy"  # will not be used

    # Environment variable to save/load checkpoints from everywhere
    if cfg.log_dir == None:
        cfg.log_dir = os.environ["LOG_DIR"]
    os.environ["LATENT_CONTROL_CKPT_DIR"] = os.path.join(cfg.log_dir, "checkpoints")

    L.seed_everything(cfg.seed, workers=True)

    # Setup checkpoint (with wandb ID as <dirpath>)
    callbacks = []
    if cfg.model_checkpoint != None:
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

    if cfg.task.dlclm != None:
        task = DLCLMTask(cfg.task.dlclm)
        cfg.task.dlclm = task.cfg

        # Add the autoencoder config to cfg
        if cfg.task.dlclm.pretrained_ae_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/dlc_lm/{cfg.task.dlclm.pretrained_ae_id}"
            )
            cfg.task.ae = OmegaConf.merge(
                OmegaConf.structured(AETaskConfig), run.config["task"]["ae"]
            )
        elif cfg.task.dlclm.pretrained_dcse_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/dlc_lm/{cfg.task.dlclm.pretrained_dcse_id}"
            )
            cfg.task.dcse = OmegaConf.merge(
                OmegaConf.structured(DCSETaskConfig),
                run.config["task"]["dcse"],
            )
    elif cfg.task.ae != None:
        task = AETask(cfg.task.ae)
        cfg.task.ae = task.cfg
    elif cfg.task.dcse != None:
        task = DCSETask(cfg.task.dcse)
        cfg.task.dcse = task.cfg
    else:
        raise ValueError("No task specified in config!")

    if cfg.compile:
        task.compile()

    # Give the whole TrainConfig to wandb
    if cfg.logger and (rank_zero_only.rank == 0):
        logger.experiment.config.update(
            OmegaConf.to_container(OmegaConf.structured(cfg)), allow_val_change=True
        )

    # Compute how many batches to accumulate to reach the effective batch size
    num_devices = (
        cfg.num_devices if cfg.num_devices != None else torch.cuda.device_count()
    )
    if cfg.effective_batch_size is None:
        accumulate_grad_batches = 1
    else:
        ddp_batch_size = task.cfg.batch_size * num_devices
        assert (
            cfg.effective_batch_size % ddp_batch_size == 0
        ), f"Effective batch size ({cfg.effective_batch_size}) must be a multiple of effective ddp batch_size ({ddp_batch_size})"
        accumulate_grad_batches = cfg.effective_batch_size // ddp_batch_size

        # just makin sure
        assert (
            accumulate_grad_batches * task.cfg.batch_size * num_devices
            == cfg.effective_batch_size
        )
        rank_zero_info(
            f"Running with {num_devices} devices, batch size {task.cfg.batch_size} per device, accumulating {accumulate_grad_batches} steps to reach effective batch size {cfg.effective_batch_size}"
        )

    # Instantiate the trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        enable_checkpointing=True if cfg.model_checkpoint else False,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval
        * accumulate_grad_batches,  # to account for accumulation
        gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
        max_steps=cfg.max_steps * accumulate_grad_batches,
        log_every_n_steps=50,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=cfg.precision,
        limit_val_batches=cfg.limit_val_batches if cfg.limit_val_batches else 1.0,
        devices=num_devices,
        strategy=cfg.strategy,
        num_nodes=1,
        use_distributed_sampler=False,
    )
    trainer.fit(
        model=task,
        ckpt_path=(
            os.path.realpath(os.path.join(cfg.model_checkpoint["dirpath"], "last.ckpt"))
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
