import os
import uuid
from typing import Optional

import numpy as np
import torch
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.similarity_functions import cos_sim, pairwise_cos_sim

from model.encoder import EncoderModel


class MTEBEncoderWrapper:
    """
    Wraps an EncoderModel for MTEB evaluation.

    Uses the SEM head output (pre-out_proj) as the embedding.
    Supports "soft" mode (standard temperature) and "hard" mode (near-zero
    temperature, producing approximately one-hot vectors).
    """

    def __init__(
        self,
        encoder: EncoderModel,
        mode: str = "soft",
        hard_temp: float = 1e-4,
        batch_size: int = 64,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        assert mode in ("soft", "hard"), f"mode must be 'soft' or 'hard', got {mode}"
        if encoder.cfg.sem is None and mode == "hard":
            raise ValueError("mode='hard' requires a SEM head (encoder.cfg.sem is None)")

        self.encoder = encoder.eval()
        self.mode = mode
        self.hard_temp = hard_temp
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

        sem = encoder.cfg.sem
        self._sem_dims = (sem["L"], sem["V"]) if sem is not None else None

        name = encoder.cfg.model_name
        if sem is not None:
            name = f"{name}-sem-L{sem['L']}-V{sem['V']}-{mode}"
        similarity_fn_name = ScoringFunction.CUSTOM if self._sem_dims is not None else "cosine"
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=name,
            revision=str(uuid.uuid4()),
            release_date=None,
            n_parameters=None,
            embed_dim=None,
            languages=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            training_datasets=None,
            similarity_fn_name=similarity_fn_name,
            memory_usage_mb=None,
            max_tokens=None,
            framework=[],
        )

    def encode(self, sentences, batch_size: int = 0, **kwargs) -> np.ndarray:
        # MTEB v2 passes a DataLoader[BatchedInput] instead of list[str]
        if not isinstance(sentences, list):
            sentences = [text for batch in sentences for text in batch["text"]]

        if batch_size == 0:
            batch_size = self.batch_size

        temp = self.hard_temp if self.mode == "hard" else None

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.encoder.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z, _ = self.encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    skip_out_proj=True,
                    temp=temp,
                )

            all_embeddings.append(z.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1, embeddings2):
        if self._sem_dims is None:
            return cos_sim(embeddings1, embeddings2)
        return self._kl_sim(embeddings1, embeddings2, pairwise=False)

    def similarity_pairwise(self, embeddings1, embeddings2):
        if self._sem_dims is None:
            return pairwise_cos_sim(embeddings1, embeddings2)
        return self._kl_sim(embeddings1, embeddings2, pairwise=True)

    def _kl_sim(self, e1, e2, pairwise: bool = False):
        """Negative symmetric KL divergence between L concatenated simplices."""
        L, V = self._sem_dims
        eps = 1e-10

        p = torch.as_tensor(np.asarray(e1), dtype=torch.float32).clamp(min=eps)
        q = torch.as_tensor(np.asarray(e2), dtype=torch.float32).clamp(min=eps)
        p = p.reshape(-1, L, V)  # (N, L, V)
        q = q.reshape(-1, L, V)  # (M, L, V)

        if pairwise:
            # Elementwise: KL(p_i || q_i) and KL(q_i || p_i), both summed over L and V
            kl_pq = (p * (p.log() - q.log())).sum(dim=(-2, -1))  # (N,)
            kl_qp = (q * (q.log() - p.log())).sum(dim=(-2, -1))  # (N,)
            return -(kl_pq + kl_qp) / 2

        # All-pairs: (N, M)
        log_p = p.log()   # (N, L, V)
        log_q = q.log()   # (M, L, V)

        p_log_p = (p * log_p).sum(dim=(-2, -1))          # (N,)
        q_log_q = (q * log_q).sum(dim=(-2, -1))          # (M,)
        p_log_q = torch.einsum("nlv, mlv -> nm", p, log_q)  # (N, M)
        q_log_p = torch.einsum("mlv, nlv -> mn", q, log_p)  # (M, N)

        kl_pq = p_log_p[:, None] - p_log_q              # (N, M)
        kl_qp = q_log_q[None, :] - q_log_p.T            # (N, M)
        return -(kl_pq + kl_qp) / 2


WANDB_PROJECT = "guillaume-lajoie/dlc_lm_4"


def load_encoder_from_checkpoint(
    wandb_id: str,
    device: str = "cuda",
) -> EncoderModel:
    """
    Load an EncoderModel from a Lightning checkpoint identified by wandb run ID.
    Automatically detects the task type (ae or declutr) from the wandb config.

    Args:
        wandb_id: The wandb run ID (used as checkpoint directory name).
        device: Device to load the model on.
    """
    import wandb as wandb_lib
    from tasks.autoencoder import AETask
    from tasks.declutr import DeCLUTRTask
    from tasks.dino_mixture import DINOMixtureTask
    from tasks.simcse import SimCSETask

    # Auto-detect task type from wandb config
    run = wandb_lib.Api().run(f"{WANDB_PROJECT}/{wandb_id}")
    config = run.config
    if "task" in config and config["task"]["ae"] is not None:
        task_type = "ae"
    elif "task" in config and config["task"]["declutr"] is not None:
        task_type = "declutr"
    elif "task" in config and config["task"]["dino_mixture"] is not None:
        task_type = "dino_mixture"
    elif "task" in config and config["task"]["simcse"] is not None:
        task_type = "simcse"
    else:
        raise ValueError(
            f"Could not detect task type from wandb config for run {wandb_id}. "
            f"Expected 'task.ae' or 'task.declutr' or 'task.dino_mixture' or 'task.simcse' in config."
        )

    print(f"Detected task type: {task_type}")

    ckpt_path = os.path.join(
        os.environ["LOG_DIR"],
        "checkpoints/",
        wandb_id,
        "last.ckpt",
    )

    if task_type == "ae":
        task = AETask.load_from_checkpoint(
            ckpt_path, strict=False, map_location=torch.device(device)
        )
    elif task_type == "dino_mixture":
        task = DINOMixtureTask.load_from_checkpoint(
            ckpt_path, strict=False, map_location=torch.device(device)
        )
        return task.teacher_encoder.eval().requires_grad_(False)
    elif task_type == "simcse":
        task = SimCSETask.load_from_checkpoint(
            ckpt_path, strict=False, map_location=torch.device(device)
        )
    else:
        task = DeCLUTRTask.load_from_checkpoint(
            ckpt_path, strict=False, map_location=torch.device(device)
        )

    return task.encoder.eval().requires_grad_(False)
