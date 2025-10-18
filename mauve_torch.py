# Author: Krishna Pillutla
# License: GPLv3

import math
import numpy as np
import time
from types import SimpleNamespace
import json
import os
import time
from tqdm.auto import tqdm as tqdm_original

import torch
from transformers import AutoModel, AutoTokenizer

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve


try:
    import torch

    FOUND_TORCH = True
except (ImportError, ModuleNotFoundError):
    FOUND_TORCH = False

try:
    import transformers

    FOUND_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    FOUND_TRANSFORMERS = False

if FOUND_TORCH and FOUND_TRANSFORMERS:
    # only needed for tokenizing
    from mauve.utils import (
        get_tokenizer,
        get_model,
        featurize_tokens_from_model,
        get_device_from_arg,
    )


MODEL, TOKENIZER, MODEL_NAME = None, None, None

CPU_DEVICE = torch.device("cpu")
tqdm = lambda *args, **kwargs: tqdm_original(
    *args, **kwargs, disable=os.environ.get("DISABLE_TQDM", False)
)


def get_device_from_arg(device_id):
    if (
        device_id is not None
        and torch.cuda.is_available()
        and 0 <= device_id < torch.cuda.device_count()
    ):
        return torch.device(f"cuda:{device_id}")
    else:
        return CPU_DEVICE


def get_model(model_name, tokenizer, device_id):
    device = get_device_from_arg(device_id)
    if "gpt2" in model_name or "bert" in model_name:
        model = AutoModel.from_pretrained(
            model_name, pad_token_id=tokenizer.eos_token_id
        ).to(device)
        model = model.eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_tokenizer(model_name="gpt2"):
    if "gpt2" in model_name or "bert" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return tokenizer


def load_json_dataset(data_path, max_num_data):
    texts = []
    for i, line in enumerate(open(data_path)):
        if i >= max_num_data:
            break
        texts.append(json.loads(line)["text"])
    return texts


def load_and_tokenize_json_data(
    tokenizer, data_path, max_len=1024, max_num_data=float("inf")
):
    """Load and tokenize the data in a jsonl format

    :param tokenizer:  HF tokenizer object
    :param data_path: jsonl file to read. Read the "text" field of each line
    :param max_len: maximum length of tokenized data
    :param max_num_data: maximum number of lines to load
    :return: list of `torch.LongTensor`s of shape (1, num_tokens), one for each input line
    """
    assert (
        max_len <= 1024 and max_num_data >= 2000
    ), f"max_len={max_len}, max_num_data={max_num_data} are insufficent"
    t1 = time.time()
    texts = load_json_dataset(data_path, max_num_data=max_num_data)
    t2 = time.time()
    print(f"dataset load time: {round(t2-t1, 2)} sec")
    t1 = time.time()
    tokenized_texts = [
        tokenizer.encode(sen, return_tensors="pt", truncation=True, max_length=max_len)
        for sen in texts
    ]
    t2 = time.time()
    print(f"tokenizing time: {round(t2-t1, 2)} sec")
    return tokenized_texts


def decode_samples_from_lst(tokenizer, tokenized_texts):
    """Decode from tokens to string

    :param tokenizer: HF tokenizer
    :param tokenized_texts: list of list of tokens
    :return: decoded output as a list of strings of the same length as tokenized_text_list
    """
    t1 = time.time()
    output = []
    for l in tokenized_texts:
        o = tokenizer.decode(torch.LongTensor(l), skip_special_tokens=True)
        output.append(o)
    t2 = time.time()
    print(f"de-tokenizing time: {round(t2-t1, 2)}")
    return output


@torch.no_grad()
def featurize_tokens_from_model(
    model, tokenized_texts, batch_size, name="", verbose=False
):
    """Featurize tokenized texts using models, support batchify
    :param model: HF Transformers model
    :param batch_size: Batch size used during forward pass
    :param tokenized_texts: list of torch.LongTensor of shape (1, length)
    :param verbose: If True, print status and time
    :return:
    """
    device = next(model.parameters()).device
    t1 = time.time()
    feats, chunks, chunk_sent_lengths = [], [], []
    chunk_idx = 0

    while chunk_idx * batch_size < len(tokenized_texts):
        _chunk = [
            _t.view(-1)
            for _t in tokenized_texts[
                chunk_idx * batch_size : (chunk_idx + 1) * batch_size
            ]
        ]
        chunks.append(_chunk)
        chunk_sent_lengths.append([len(_c) for _c in _chunk])
        chunk_idx += 1

    for chunk, chunk_sent_length in tqdm(
        list(zip(chunks, chunk_sent_lengths)), desc=f"Featurizing {name}"
    ):
        padded_chunk = torch.nn.utils.rnn.pad_sequence(
            chunk, batch_first=True, padding_value=0
        ).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(sent_length).long() for sent_length in chunk_sent_length],
            batch_first=True,
            padding_value=0,
        ).to(device)
        outs = model(
            input_ids=padded_chunk,
            attention_mask=attention_mask,
            past_key_values=None,
            output_hidden_states=True,
            return_dict=True,
        )
        h = []
        for hidden_state, sent_length in zip(outs.hidden_states[-1], chunk_sent_length):
            h.append(hidden_state[sent_length - 1])
        h = torch.stack(h, dim=0)
        feats.append(h.cpu())
    t2 = time.time()
    if verbose:
        print(f"Featurize time: {round(t2-t1, 2)}")
    return torch.cat(feats)


def compute_mauve(
    p_features=None,
    q_features=None,
    p_tokens=None,
    q_tokens=None,
    p_text=None,
    q_text=None,
    num_buckets="auto",
    pca_max_data=-1,
    kmeans_explained_var=0.9,
    kmeans_num_redo=5,
    kmeans_max_iter=500,
    featurize_model_name="gpt2-large",
    device_id=-1,
    max_text_length=1024,
    divergence_curve_discretization_size=25,
    mauve_scaling_factor=5,
    verbose=False,
    seed=25,
    batch_size=1,
    use_float64=False,
):
    """
    Compute the MAUVE score between two text generations P and Q.

    P is either specified as ``p_features``, ``p_tokens``, or ``p_text``. Same with Q.

    :param ``p_features``: ``numpy.ndarray`` of shape (n, d), where n is the number of generations.
    :param ``q_features``: ``numpy.ndarray`` of shape (n, d), where n is the number of generations.
    :param ``p_tokens``: list of length n, each entry is torch.LongTensor of shape (1, length).
    :param ``q_tokens``: list of length n, each entry is torch.LongTensor of shape (1, length).
    :param ``p_text``: list of length n, each entry is a string.
    :param ``q_text``: list of length n, each entry is a string.
    :param ``num_buckets``: the size of the histogram to quantize P and Q. Options: ``'auto'`` (default, which is n/10) or an integer.
    :param ``pca_max_data``: the number data points to use for PCA. If `-1`, use all the data. Default -1.
    :param ``kmeans_explained_var``: amount of variance of the data to keep in dimensionality reduction by PCA. Default 0.9.
    :param ``kmeans_num_redo``: number of times to redo k-means clustering (the best objective is kept). Default 5.
        Try reducing this to 1 in order to reduce running time.
    :param ``kmeans_max_iter``: maximum number of k-means iterations. Default 500.
        Try reducing this to 100 in order to reduce running time.
    :param ``featurize_model_name``: name of the model from which features are obtained. Default 'gpt2-large'.
        We support all models which can be loaded from ``transformers.AutoModel.from_pretrained(featurize_model_name)``.
    :param ``device_id``: Device for featurization. Supply gpu_id (e.g. 0 or 3) to use GPU or -1 to use CPU.
    :param ``max_text_length``: maximum number of tokens to consider. Default 1024.
    :param ``divergence_curve_discretization_size``: Number of points to consider on the divergence curve. Default 25.
        Larger values do not offer much of a difference.
    :param ``mauve_scaling_factor``: The constant``c`` from the paper. Default 5.
        See `Best Practices <index.html#best-practices-for-mauve>`_ for details.
    :param ``verbose``: If True, print running time updates.
    :param ``seed``: random seed to initialize k-means cluster assignments.
    :param ``batch_size``: Batch size for feature extraction

    :return: an object with fields p_hist, q_hist, divergence_curve and mauve.

    * ``out.mauve`` is a number between 0 and 1, the MAUVE score. Higher values means P is closer to Q.
    * ``out.frontier_integral``, a number between 0 and 1. Lower values mean that P is closer to Q.
    * ``out.p_hist`` is the obtained histogram for P. Same for ``out.q_hist``.
    * ``out.divergence_curve`` contains the points in the divergence curve. It is of shape (m, 2), where m is ``divergence_curve_discretization_size``

    """

    if p_features is None and p_tokens is None and p_text is None:
        raise ValueError("Supply at least one of p_features, p_tokens, p_text")
    if q_features is None and q_tokens is None and q_text is None:
        raise ValueError("Supply at least one of q_features, q_tokens, q_text")
    p_features = get_features_from_input(
        p_features,
        p_tokens,
        p_text,
        featurize_model_name,
        max_text_length,
        device_id,
        name="p",
        verbose=verbose,
        batch_size=batch_size,
        use_float64=use_float64,
    )
    q_features = get_features_from_input(
        q_features,
        q_tokens,
        q_text,
        featurize_model_name,
        max_text_length,
        device_id,
        name="q",
        verbose=verbose,
        batch_size=batch_size,
        use_float64=use_float64,
    )
    if num_buckets == "auto":
        # heuristic: use num_clusters = num_generations / 10
        num_buckets = max(
            2, int(round(min(p_features.shape[0], q_features.shape[0]) / 10))
        )
    elif not isinstance(num_buckets, int):
        raise ValueError('num_buckets is expected to be an integer or "auto"')

    # Acutal binning
    t1 = time.time()
    p, q = cluster_feats(
        p_features,
        q_features,
        num_clusters=num_buckets,
        norm="l2",
        whiten=False,
        pca_max_data=pca_max_data,
        explained_variance=kmeans_explained_var,
        num_redo=kmeans_num_redo,
        max_iter=kmeans_max_iter,
        seed=seed,
        verbose=verbose,
    )
    t2 = time.time()
    if verbose:
        print("total discretization time:", round(t2 - t1, 2), "seconds")

    # Divergence curve and mauve
    mixture_weights = np.linspace(1e-6, 1 - 1e-6, divergence_curve_discretization_size)
    divergence_curve = get_divergence_curve_for_multinomials(
        p, q, mixture_weights, mauve_scaling_factor
    )
    x, y = divergence_curve.T
    idxs1 = np.argsort(x)
    idxs2 = np.argsort(y)
    mauve_score = 0.5 * (
        compute_area_under_curve(x[idxs1], y[idxs1])
        + compute_area_under_curve(y[idxs2], x[idxs2])
    )
    fi_score = get_fronter_integral(p, q)
    to_return = SimpleNamespace(
        p_hist=p,
        q_hist=q,
        divergence_curve=divergence_curve,
        mauve=mauve_score,
        frontier_integral=fi_score,
        num_buckets=num_buckets,
    )
    return to_return


def get_features_from_input(
    features,
    tokenized_texts,
    texts,
    featurize_model_name,
    max_len,
    device_id,
    name,
    batch_size,
    verbose=False,
    use_float64=False,
):
    global MODEL, TOKENIZER, MODEL_NAME
    if features is None:
        # Featurizing is necessary. Make sure the required packages are available
        if not FOUND_TORCH:
            raise ModuleNotFoundError(
                """PyTorch not found. Please install PyTorch if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve` 
                    and `https://pytorch.org/get-started/locally/`.
                """
            )
        if not FOUND_TRANSFORMERS:
            raise ModuleNotFoundError(
                """Transformers not found. Please install Transformers if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve` 
                    and `https://huggingface.co/transformers/installation.html`.
                """
            )

        if tokenized_texts is None:
            # tokenize texts
            if TOKENIZER is None or MODEL_NAME != featurize_model_name:
                if verbose:
                    print("Loading tokenizer")
                TOKENIZER = get_tokenizer(featurize_model_name)
            if verbose:
                print("Tokenizing text...")
            tokenized_texts = [
                TOKENIZER.encode(
                    sen, return_tensors="pt", truncation=True, max_length=max_len
                )
                for sen in texts
            ]
        # use tokenized_texts to featurize
        if TOKENIZER is None or MODEL_NAME != featurize_model_name:
            if verbose:
                print("Loading tokenizer")
            TOKENIZER = get_tokenizer(featurize_model_name)
        if MODEL is None or MODEL_NAME != featurize_model_name:
            if verbose:
                print("Loading model")
            MODEL = get_model(featurize_model_name, TOKENIZER, device_id)
            MODEL_NAME = featurize_model_name
        else:
            MODEL = MODEL.to(get_device_from_arg(device_id))
        if use_float64:
            MODEL = MODEL.double()
        if verbose:
            print("Featurizing tokens")
        features = (
            featurize_tokens_from_model(MODEL, tokenized_texts, batch_size, name)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        features = np.asarray(features)
    return features


def cluster_feats(
    p,
    q,
    num_clusters,
    norm="none",
    whiten=True,
    pca_max_data=-1,
    explained_variance=0.9,
    num_redo=5,
    max_iter=500,
    seed=0,
    verbose=False,
):
    assert 0 < explained_variance < 1
    if verbose:
        print(f"seed = {seed}")
    assert norm in ["none", "l2", "l1", None]
    data1 = np.vstack([q, p])
    if norm in ["l2", "l1"]:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed + 5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(
            f"Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints"
        )
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    if verbose:
        print(f"performing clustering in lower dimension = {idx}")
    data1 = pca.transform(data1)[:, : idx + 1]
    # Cluster
    data1 = data1.astype(np.float32)
    """
    t1 = time.time()
    print(f"k-means by faiss")
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          verbose=verbose, nredo=num_redo, update_index=True,
                          seed=seed+2, gpu=True)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    t2 = time.time()
    if verbose:
        print('kmeans time:', round(t2-t1, 2), 's') # 668.27 s
    """
    t1 = time.time()
    print(f"k-means by me")
    kmeans = KMeans(n_clusters=num_clusters, n_init=num_redo, max_iter=max_iter)
    clusters = kmeans.fit(torch.from_numpy(data1).cuda())
    labels = [None] * data1.shape[0]
    for i, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = i
    labels = np.array(labels)
    t2 = time.time()
    if verbose:
        print("kmeans time:", round(t2 - t1, 2), "s")  # 0.65 s

    q_labels = labels[: len(q)]
    p_labels = labels[len(q) :]

    q_bins = np.histogram(
        q_labels, bins=num_clusters, range=[0, num_clusters], density=True
    )[0]
    p_bins = np.histogram(
        p_labels, bins=num_clusters, range=[0, num_clusters], density=True
    )[0]
    return p_bins / p_bins.sum(), q_bins / q_bins.sum()


def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_divergence_curve_for_multinomials(p, q, mixture_weights, scaling_factor):
    # TODO: check if extreme points are needed
    divergence_curve = [[0, np.inf]]  # extreme point
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0])  # other extreme point
    return np.exp(-scaling_factor * np.asarray(divergence_curve))


def get_fronter_integral(p, q, scaling_factor=2):
    total = 0.0
    for p1, q1 in zip(p, q):
        if p1 == 0 and q1 == 0:
            pass
        elif p1 == 0:
            total += q1 / 4
        elif q1 == 0:
            total += p1 / 4
        elif abs(p1 - q1) > 1e-8:
            t1 = p1 + q1
            t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
            total += 0.25 * t1 - 0.5 * t2
        # else: contribution is 0
    return total * scaling_factor


# k-means through pytorch on cuda
class KMeans:
    def __init__(
        self, n_clusters, n_init=10, max_iter=300, min_variation=1e-3, device="cuda"
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.min_variation = min_variation
        self.device = device

    def fit(self, encodings):
        if self.n_clusters < 1:
            raise Exception(
                f"the number of clusters should be >=1, but got {self.n_clusters}"
            )
        if self.n_clusters == 1:
            clusters = [list(range(encodings.shape[0]))]
        else:
            best_group_index, min_loss = None, 1e10
            for init in range(self.n_init):
                loss = np.nan
                while np.isnan(loss):
                    group_index, loss = self.fit_once(encodings, init)
                if loss < min_loss:
                    best_group_index = group_index
            clusters = [[] for _ in range(self.n_clusters)]
            for i, index in enumerate(best_group_index.tolist()):
                clusters[index].append(i)
        return clusters

    @torch.no_grad()
    def fit_once(self, encodings, init):
        encodings = torch.nn.functional.normalize(encodings.to(self.device), dim=-1)

        unique_encodings = torch.unique(encodings, dim=0)
        if unique_encodings.shape[0] < self.n_clusters:
            self.n_clusters = unique_encodings.shape[0]
        centers = None
        ceil = torch.Tensor([1.0]).to(self.device)
        for i, idx in enumerate(torch.randperm(unique_encodings.shape[0]).tolist()):
            if i == 0:
                centers = unique_encodings[idx].unsqueeze(0)
                continue
            new_center = unique_encodings[idx].unsqueeze(0)
            if not torch.isclose(torch.mm(centers, new_center.T).max(), ceil):
                centers = torch.cat([centers, new_center], dim=0)
                if centers.shape[0] == self.n_clusters:
                    break

        from tqdm import tqdm

        with tqdm(total=self.max_iter, desc=f"KMeans ({init+1}/{self.n_init})") as bar:
            for iter_step in range(self.max_iter):
                old_centers = centers
                group_index, loss = self.group_points(centers, encodings)
                centers = self.update_centers(group_index, encodings, old_centers)
                centers_max_movement = (
                    ((old_centers - centers) ** 2).sum(dim=-1).max().item()
                )
                bar.set_description(
                    f"KMeans ({init+1}/{self.n_init}): "
                    f"loss: {loss:.3f} | "
                    f"movement: {centers_max_movement:.3f}"
                )
                if centers_max_movement < self.min_variation or np.isnan(loss):
                    bar.total = iter_step + 1
                    bar.update(1)
                    break
                else:
                    bar.update(1)
        return group_index, loss

    @torch.no_grad()
    def group_points(self, centers, encodings, capacity=int(1e10)):
        # centers: [n_clusters, hs]
        # encodings: [N, hs]
        split_len = capacity // (encodings.shape[0] * centers.shape[0])
        split_num = np.math.ceil(encodings.shape[0] / split_len)
        group_index = []
        loss = 0.0
        for i in range(split_num):
            split_encodings = encodings[i * split_len : (i + 1) * split_len, :]
            split_distances = 1 - torch.mm(split_encodings, centers.T)
            group_index.append(split_distances.argmin(dim=-1).detach())
            loss += split_distances.min(dim=-1).values.sum().item()
        group_index = torch.cat(group_index, dim=0)
        return group_index, loss  # [n_clusters], float

    @torch.no_grad()
    def update_centers(self, group_index, encodings, old_centers):
        # sum_vec = torch.zeros([self.n_clusters, encodings.shape[1]],
        #                      dtype=encodings.dtype, device=self.device)
        sum_vec = old_centers * 1e-6
        count_vec = torch.zeros([self.n_clusters], device=self.device) + 1e-6
        index = group_index.unsqueeze(1).repeat(
            1, encodings.shape[1]
        )  # [n_clusters, hs]
        sum_vec = sum_vec.scatter_add_(dim=0, index=index, src=encodings)
        count_vec = count_vec.scatter_add_(
            dim=0, index=group_index, src=torch.ones_like(group_index).float()
        )
        mean_vec = sum_vec.div_(count_vec.unsqueeze(1))
        centers = torch.nn.functional.normalize(mean_vec, dim=-1)
        return centers