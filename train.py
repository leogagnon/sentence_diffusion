import os
import logging as py_logging

# Must be set before importing huggingface_hub, transformers, or datasets,
# as they read these at module import time.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch._dynamo.config.capture_scalar_outputs = True

import argparse
import subprocess
from dataclasses import dataclass
from typing import Optional

import hydra
import lightning as L
import wandb
from hydra.core.config_store import ConfigStore
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from omegaconf import OmegaConf, SCMode
from transformers.utils import logging

from tasks.autoencoder import AETask, AETaskConfig
from tasks.declutr import DeCLUTRTask, DeCLUTRTaskConfig
from tasks.dlc_ar import DLCARTask, DLCARTaskConfig
from tasks.dlc_md import DLCMDTask, DLCMDTaskConfig
from tasks.ddpm import GaussianDiffusionTask, GaussianDiffusionTaskConfig
from tasks.dino_mixture import DINOMixtureTask, DINOMixtureTaskConfig
from tasks.simcse import SimCSETask, SimCSETaskConfig
from tasks.supervised_simcse import SupervisedSimCSETask, SupervisedSimCSETaskConfig
from tasks.distill_simcse import DistillSimCSETask, DistillSimCSETaskConfig

logging.set_verbosity_error()
for datasets_logger_name in [
    "datasets",
    "datasets.load",
    "datasets.packaged_modules.cache.cache",
]:
    py_logging.getLogger(datasets_logger_name).setLevel(py_logging.ERROR)

torch.set_float32_matmul_precision("medium")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONFAULTHANDLER"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_DISABLE_ADDR2LINE"] = "1"
os.environ["TORCH_FR_BUFFER_SIZE"] = "1024"

def infer_num_nodes(default=1):
    # torchrun / Lightning envs
    world_size = int(os.getenv("WORLD_SIZE", "0"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0"))
    if world_size > 0 and local_world_size > 0:
        return max(1, world_size // local_world_size)

    # Slurm fallback
    if os.getenv("SLURM_NNODES"):
        return int(os.environ["SLURM_NNODES"])
    if os.getenv("SLURM_JOB_NUM_NODES"):
        return int(os.environ["SLURM_JOB_NUM_NODES"])

    return default

@dataclass
class TaskConfig:
    ae: Optional[AETaskConfig] = None
    declutr: Optional[DeCLUTRTaskConfig] = None
    dlc_ar: Optional[DLCARTaskConfig] = None
    dlc_md: Optional[DLCMDTaskConfig] = None
    dlc_ddpm: Optional[GaussianDiffusionTaskConfig] = None
    dino_mixture: Optional[DINOMixtureTaskConfig] = None
    simcse: Optional[SimCSETaskConfig] = None
    supervised_simcse: Optional[SupervisedSimCSETaskConfig] = None
    distill_simcse: Optional[DistillSimCSETaskConfig] = None


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
    workers: Optional[int] = None
    initial_val: bool = False


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
        project = "ul_text"
        run = api.run(f"{entity}/{project}/{run_id}")
        cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), run.config)

        # Set the logger ID to the provided run_id (to resume the run)
        cfg.logger.id = wandb_id
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        # Setup tags for wandb
        tags = [k for k in cfg.task.keys() if cfg.task[k] is not None]
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
    if cfg.log_dir is None:
        cfg.log_dir = os.environ["LOG_DIR"]
    os.environ["LATENT_CONTROL_CKPT_DIR"] = os.path.join(cfg.log_dir, "checkpoints")

    if cfg.workers is None:
        cfg.workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    os.environ["TORCH_NUM_WORKERS"] = str(cfg.workers)

    rank_zero_info(f"Using {cfg.workers} dataloader workers per process.")
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

    if cfg.task.dlc_ar != None:
        task = DLCARTask(cfg.task.dlc_ar)
        cfg.task.dlc_ar = task.cfg

        # Add the autoencoder config to cfg
        if cfg.task.dlc_ar.pretrained_ae_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/ul_text/{cfg.task.dlc_ar.pretrained_ae_id}"
            )
            cfg.task.ae = OmegaConf.merge(
                OmegaConf.structured(AETaskConfig), run.config["task"]["ae"]
            )
        elif cfg.task.dlc_ar.pretrained_declutr_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/ul_text/{cfg.task.dlc_ar.pretrained_declutr_id}"
            )
            cfg.task.declutr = OmegaConf.merge(
                OmegaConf.structured(DeCLUTRTaskConfig), run.config["task"]["declutr"]
            )

    elif cfg.task.dlc_md != None:
        task = DLCMDTask(cfg.task.dlc_md)
        cfg.task.dlc_md = task.cfg

        # Add the autoencoder config to cfg
        if cfg.task.dlc_md.pretrained_ae_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/ul_text/{cfg.task.dlc_md.pretrained_ae_id}"
            )
            cfg.task.ae = OmegaConf.merge(
                OmegaConf.structured(AETaskConfig), run.config["task"]["ae"]
            )
        elif cfg.task.dlc_md.pretrained_declutr_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/ul_text/{cfg.task.dlc_md.pretrained_declutr_id}"
            )
            cfg.task.declutr = OmegaConf.merge(
                OmegaConf.structured(DeCLUTRTaskConfig), run.config["task"]["declutr"]
            )
    elif cfg.task.dlc_ddpm != None:
        task = GaussianDiffusionTask(cfg.task.dlc_ddpm)
        cfg.task.dlc_ddpm = task.cfg

        if cfg.task.dlc_ddpm.pretrained_ae_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/ul_text/{cfg.task.dlc_ddpm.pretrained_ae_id}"
            )
            cfg.task.ae = OmegaConf.merge(
                OmegaConf.structured(AETaskConfig), run.config["task"]["ae"]
            )
        elif cfg.task.dlc_ddpm.pretrained_declutr_id is not None:
            run = wandb.Api().run(
                f"guillaume-lajoie/ul_text/{cfg.task.dlc_ddpm.pretrained_declutr_id}"
            )
            cfg.task.declutr = OmegaConf.merge(
                OmegaConf.structured(DeCLUTRTaskConfig), run.config["task"]["declutr"]
            )
    elif cfg.task.ae != None:
        task = AETask(cfg.task.ae)
        cfg.task.ae = task.cfg
    elif cfg.task.declutr != None:
        task = DeCLUTRTask(cfg.task.declutr)
        cfg.task.declutr = task.cfg
    elif cfg.task.dino_mixture != None:
        task = DINOMixtureTask(cfg.task.dino_mixture)
        cfg.task.dino_mixture = task.cfg
    elif cfg.task.simcse != None:
        task = SimCSETask(cfg.task.simcse)
        cfg.task.simcse = task.cfg
    elif cfg.task.supervised_simcse != None:
        task = SupervisedSimCSETask(cfg.task.supervised_simcse)
        cfg.task.supervised_simcse = task.cfg
    elif cfg.task.distill_simcse != None:
        task = DistillSimCSETask(cfg.task.distill_simcse)
        cfg.task.distill_simcse = task.cfg
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

    num_nodes = infer_num_nodes()

    ddp_batch_size = task.cfg.batch_size * num_devices * num_nodes

    if cfg.effective_batch_size is None:
        accumulate_grad_batches = 1
        cfg.effective_batch_size = ddp_batch_size
    else:
        assert (
            cfg.effective_batch_size % ddp_batch_size == 0
        ), f"Effective batch size ({cfg.effective_batch_size}) must be a multiple of effective ddp batch_size ({ddp_batch_size})"
        accumulate_grad_batches = cfg.effective_batch_size // ddp_batch_size

        # just makin sure
        assert (
            accumulate_grad_batches * task.cfg.batch_size * num_devices * num_nodes
            == cfg.effective_batch_size
        )
    rank_zero_info(
        f"Running with {num_devices} devices per node, {num_nodes} nodes, batch size {task.cfg.batch_size} per device, accumulating {accumulate_grad_batches} steps to reach effective batch size {cfg.effective_batch_size}"
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
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        max_steps=cfg.max_steps,
        log_every_n_steps=50,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=cfg.precision,
        limit_val_batches=cfg.limit_val_batches if cfg.limit_val_batches else 1.0,
        devices=num_devices,
        strategy=cfg.strategy,
        num_nodes=num_nodes,
        use_distributed_sampler=False
    )
    if cfg.initial_val:
        trainer.validate(task)
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
