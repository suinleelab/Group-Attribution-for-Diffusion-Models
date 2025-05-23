"""
evaluation script for the precision-recall metric proposed by Kynkäänniemi et al. [^1]
the implementation is based on the code of stylegan2-ada-pytorch [^2]

[^1]: Kynkäänniemi, Tuomas, et al. "Improved precision and recall metric for assessing generative models." Advances in Neural Information Processing Systems 32 (2019).
[^2]: https://github.com/NVlabs/stylegan2-ada-pytorch
"""  # noqa

import math
import os
from collections import namedtuple
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import src.constants as constants
from src.datasets import ImageDataset, create_dataset

Manifold = namedtuple("Manifold", ["features", "kth"])


class VGGFeatureExtractor(nn.Module):
    """VGG network to extract features"""

    WEIGHTS_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"

    def __init__(self):
        super().__init__()
        self.model = self._load_model()

    def _load_model(self):
        model_path = os.path.join(
            constants.PRETRAINEDMODEL_DIR,
            os.path.basename(self.WEIGHTS_URL),
        )
        if not os.path.exists(model_path):
            download_url_to_file(self.WEIGHTS_URL, model_path)
        model = torch.jit.load(model_path).eval()
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
        return model

    def forward(self, x):
        return self.model(x, return_features=True)


def compute_distance(
    row_features, col_features, row_batch_size, col_batch_size, device
):
    """Compute distance between row and col features."""
    dist = []
    for row_batch in row_features.split(row_batch_size, dim=0):
        dist_batch = []
        for col_batch in col_features.split(col_batch_size, dim=0):
            dist_batch.append(
                torch.cdist(
                    row_batch.to(device).unsqueeze(0), col_batch.to(device).unsqueeze(0)
                )
                .squeeze(0)
                .cpu()
            )
        dist_batch = torch.cat(dist_batch, dim=1)
        dist.append(dist_batch)
    dist = torch.cat(dist, dim=0)
    return dist


def to_uint8(x):
    """Convert x to RGB scale and torch unit8."""
    return (x * 127.5 + 128).clamp(0, 255).to(torch.uint8)


class ManifoldBuilder:
    """Initialize data manifold for a given dataset"""

    def __init__(
        self,
        data=None,
        model=None,
        features=None,
        extr_batch_size=128,
        max_sample_size=50000,
        nhood_size=3,
        row_batch_size=10000,
        col_batch_size=10000,
        random_state=1234,
        num_workers=0,
        device=torch.device("cpu"),  # set to cuda if available for the best performance
    ):
        if features is None:
            num_extr_batches = math.ceil(max_sample_size / extr_batch_size)
            if model is None:
                if hasattr(data, "__getitem__") and hasattr(
                    data, "__len__"
                ):  # map-style dataset
                    data_size = len(data)
                    if data_size > max_sample_size:
                        np.random.seed(random_state)
                        inds = torch.as_tensor(
                            np.random.choice(
                                data_size, size=max_sample_size, replace=False
                            )
                        )
                        data = Subset(data, indices=inds)

                    def dataloader():
                        _dataloader = DataLoader(
                            data,
                            batch_size=extr_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False,
                            pin_memory=True,
                        )
                        for x in _dataloader:
                            if isinstance(x, (list, tuple)):
                                yield x[0]
                            else:
                                yield x

                else:
                    assert isinstance(data, (np.ndarray, torch.Tensor, str))
                    if isinstance(data, str) and os.path.exists(data):
                        fmt = data.split(".")[-1]
                        if fmt == "npy":
                            data = np.load(data)
                        elif fmt == "pt":
                            data = torch.load(data)
                    data = torch.as_tensor(data)
                    assert data.dtype == torch.uint8
                    data_size = data.shape[0]
                    if data_size > max_sample_size:
                        np.random.seed(random_state)
                        inds = torch.as_tensor(
                            np.random.choice(
                                data_size, size=max_sample_size, replace=False
                            )
                        )
                        data = data[inds]

                    def dataloader():
                        for i in range(num_extr_batches):
                            if i == num_extr_batches - 1:
                                yield data[i * extr_batch_size : max_sample_size]
                            else:
                                yield data[
                                    i * extr_batch_size : (i + 1) * extr_batch_size
                                ]

            else:

                def dataloader():
                    for i in range(num_extr_batches):
                        if i == num_extr_batches - 1:
                            yield to_uint8(
                                model.sample_x(max_sample_size - extr_batch_size * i)
                            )
                        else:
                            yield to_uint8(model.sample_x(extr_batch_size))

            self.op_device = input_device = device
            if isinstance(device, list):
                self.extractor = nn.DataParallel(
                    VGGFeatureExtractor().to(device[0]), device_ids=device
                )
                self.op_device = device[0]
                input_device = "cpu"
            else:
                self.extractor = VGGFeatureExtractor().to(device)

            features = []
            with torch.inference_mode():
                for x in tqdm(
                    dataloader(), desc="Extracting features", total=num_extr_batches
                ):
                    features.append(self.extractor(x.to(input_device)).cpu())
            features = torch.cat(features, dim=0)
        else:
            assert isinstance(features, torch.Tensor) and features.grad_fn is None
        features = features.to(
            torch.float16
        )  # half precision for faster distance computation?

        self.nhood_size = nhood_size
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.device = device

        self.features = features
        self.kth = self.compute_kth(features)

    def compute_distance(self, row_features, col_features):
        return compute_distance(
            row_features,
            col_features,
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            device=self.op_device,
        )

    def compute_kth(
        self, row_features: torch.Tensor, col_features: torch.Tensor = None
    ):
        if col_features is None:
            col_features = row_features
        kth = []

        for row_batch in tqdm(
            row_features.split(self.row_batch_size, dim=0), desc="Computing k-th radii"
        ):
            dist_batch = self.compute_distance(
                row_features=row_batch, col_features=col_features
            )
            kth.append(
                dist_batch.to(torch.float32)
                .kthvalue(self.nhood_size + 1, dim=1)
                .values.to(torch.float16)
            )  # nhood_size + 1 to exclude itself
        kth = torch.cat(kth)
        return kth

    def save(self, fpath):
        save_dir = os.path.dirname(fpath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.manifold, fpath)

    @property
    def manifold(self):
        return Manifold(features=self.features, kth=self.kth)


def calc_pr(
    manifold_1: Manifold,
    manifold_2: Manifold,
    row_batch_size: int,
    col_batch_size: int,
    device,
):
    """
    Args:
    ----

        manifold_1: generated manifold namedtuple
        (support points, radii of k-th neighborhood (inclusive))
        manifold_2: ground truth manifold namedtuple
        (support points, radii of k-th neighborhood (inclusive))
        row_batch_size: literally
        col_batch_size: literally

    Returns
    -------
        precision and recall
    """
    # ======= precision ======= #
    pred = []
    for probe_batch in tqdm(
        manifold_1.features.split(row_batch_size), desc="Calculating precision"
    ):
        dist_batch = compute_distance(
            probe_batch,
            manifold_2.features,
            row_batch_size=row_batch_size,
            col_batch_size=col_batch_size,
            device=device,
        )
        pred.append((dist_batch <= manifold_2.kth.unsqueeze(0)).any(dim=1))
    precision = torch.cat(pred).to(torch.float32).mean()

    # ======= recall ======= #
    pred.clear()
    for probe_batch in tqdm(
        manifold_2.features.split(row_batch_size), desc="Calculating recall"
    ):
        dist_batch = compute_distance(
            probe_batch,
            manifold_1.features,
            row_batch_size=row_batch_size,
            col_batch_size=col_batch_size,
            device=device,
        )
        pred.append((dist_batch <= manifold_1.kth.unsqueeze(0)).any(dim=1))
    recall = torch.cat(pred).to(torch.float32).mean()

    precision = precision.detach().cpu().item()
    recall = recall.detach().cpu().item()

    return precision, recall


def eval_pr(
    dataset,
    images,
    batch_size,
    row_batch_size=10000,
    col_batch_size=10000,
    nhood_size=3,
    device="cuda:0",
    reference_dir=None,
):
    """Evaluate precision and recall for a given dataset and a reference set"""
    eval_total_size = 50000
    # decimal_places = math.ceil(math.log(eval_total_size, 10))

    _ManifoldBuilder = partial(
        ManifoldBuilder,
        extr_batch_size=batch_size,
        max_sample_size=eval_total_size,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        nhood_size=nhood_size,
        num_workers=1,
        device=device,
    )

    gen_manifold = deepcopy(_ManifoldBuilder(data=images).manifold)

    if reference_dir is not None:
        # If reference dir exists
        manifold_path = os.path.join(reference_dir, f"pr_manifold_{dataset}.pt")
        if not os.path.exists(manifold_path):

            imageloader = ImageDataset(reference_dir)
            manifold_builder = _ManifoldBuilder(data=imageloader)
            manifold_builder.save(manifold_path)
            true_manifold = deepcopy(manifold_builder.manifold)
            del manifold_builder
        else:
            true_manifold = torch.load(manifold_path)
    else:
        # using args.dataset as default reference sets.

        manifold_path = os.path.join(
            constants.DATASET_DIR, dataset, "manifold", f"pr_manifold_{dataset}.pt"
        )
        if os.path.exists(manifold_path):
            true_manifold = torch.load(manifold_path)
        else:
            manifold_builder = _ManifoldBuilder(
                data=create_dataset(dataset_name=dataset, train=True)
            )
            manifold_builder.save(manifold_path)
            true_manifold = deepcopy(manifold_builder.manifold)
            del manifold_builder

    precision, recall = calc_pr(
        gen_manifold,
        true_manifold,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        device=device,
    )

    return precision, recall
