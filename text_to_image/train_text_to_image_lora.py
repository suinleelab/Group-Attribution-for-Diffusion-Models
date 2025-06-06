# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File adapted from https://github.com/huggingface/diffusers/blob/v0.24.0-release/examples/text_to_image/train_text_to_image_lora.py
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import sys
import random
import shutil
from pathlib import Path

import datasets
import diffusers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import time
from src.datasets import (
    remove_data_by_shapley,
    remove_data_by_uniform,
    remove_data_by_datamodel,
    remove_data_by_loo,
    remove_data_for_aoi,
)
from src.ddpm_config import PromptConfig, LoraUnlearningConfig, LoraSparseUnlearningConfig
from src.utils import fix_get_processor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")


# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


def save_model_card(
    repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=6,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="retrain",
        choices=["retrain", "pruned_ft", "sparse_gd", "gd"],
        help="training or unlearning method",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        help="ratio for remaining parameters",
        default=0.3,
    )
    parser.add_argument(
        "--checkpoint_attn_procs",
        action="store_true",
        help="whether or not to save attention processors when checkpointing",
    )
    parser.add_argument(
        "--cls_key",
        type=str,
        default=None,
        help="dataset key for class labels",
    )
    parser.add_argument(
        "--cls",
        type=str,
        default=None,
        help="fine-tune only on a specific class in the dataset",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["uniform", "shapley", "datamodel", "loo", "aoi"],
        default=None,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="alpha value for the datamodel removal distribution",
        default=None,
    )
    parser.add_argument(
        "--loo_idx",
        type=int,
        help="index to remove for the leave-one-out distribution",
        default=None,
    )
    parser.add_argument(
        "--aoi_idx",
        type=int,
        help="index to add for the add-one-in distribution",
        default=None,
    )
    parser.add_argument(
        "--removal_rank_file",
        type=str,
        help="numpy file containing top units to remove",
        default=None,
    )
    parser.add_argument(
        "--removal_rank_proportion",
        type=float,
        help="proportion of top ranked units to remove",
        default=None,
    )
    parser.add_argument(
        "--removal_bottom_proportion",
        type=float,
        help="proportion of bottom ranked units to remove",
        default=None,
    )
    parser.add_argument(
        "--removal_unit",
        type=str,
        help="unit of data for removal",
        choices=["artist", "filename"],
        default=None,
    )
    parser.add_argument(
        "--removal_seed",
        type=int,
        help="random seed for sampling from the removal distribution",
        default=0,
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        help="directory containing trained LoRA weights to load",
        default=None,
    )
    parser.add_argument(
        "--lora_steps",
        type=int,
        help="number of trained steps for the LoRA weights to load",
        default=None,
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    removal_dir = "full"
    if args.removal_dist is not None:
        if args.removal_unit is None:
            raise ValueError("--removal_unit is not specified")
        
        removal_dir_dist = args.removal_dist
        if args.removal_dist == "datamodel":
            removal_dir_dist += f"_alpha={args.datamodel_alpha}"
        
        removal_dir = f"{args.removal_unit}_{removal_dir_dist}"
        if args.removal_dist == "loo":
            removal_dir += f"/{removal_dir_dist}_idx={args.loo_idx}"
        elif args.removal_dist == "aoi":
            removal_dir += f"/{removal_dir_dist}_idx={args.aoi_idx}"
        else:
            removal_dir += f"/{removal_dir_dist}_seed={args.removal_seed}"
    
    if args.removal_rank_file is not None:
        if args.removal_rank_proportion is not None:
            removal_dir = f"counterfactual_top_{args.removal_rank_proportion}"
        elif args.removal_bottom_proportion is not None:
            removal_dir = f"counterfactual_bottom_{args.removal_bottom_proportion}"
        else:
            raise ValueError
        rank_method = os.path.basename(args.removal_rank_file).split(".")[0]
        removal_dir += f"/{rank_method}"

    assert args.dataset_name is None
    args.dataset = (
        "artbench" if "artbench" in args.train_data_dir else args.train_data_dir
    )
    if args.cls is not None and args.cls_key is not None:
        args.dataset = args.dataset + f"_{args.cls}"
    args.model_outdir, args.sample_outdir = None, None

    expected_num_lora_params = None
    if args.method == "pruned_ft":
        args.method = f"pruned_ft_ratio={args.pruning_ratio}_lr={args.learning_rate}"
        args.lora_dir = os.path.join(
            args.output_dir,
            args.dataset,
            f"pruned_ratio={args.pruning_ratio}",
            "models",
            removal_dir,
        )
        info_df = pd.read_csv(os.path.join(args.lora_dir, "info.csv"))
        expected_num_lora_params = info_df[info_df["metric"] == "pruned_lora_params"]
        expected_num_lora_params = expected_num_lora_params["value"].item()
    elif args.method in ["sparse_gd"]:
        if args.dataset == "artbench_post_impressionism":
            sparse_unlearning_config = LoraSparseUnlearningConfig.artbench_post_impressionism_config
        else:
            raise NotImplementedError
        args.lora_dir = sparse_unlearning_config["lora_dir"]
        args.lora_steps = sparse_unlearning_config["lora_steps"]
        args.max_train_steps = sparse_unlearning_config["max_train_steps"]
    elif args.method in ["gd"]:
        if args.dataset == "artbench_post_impressionism":
            unlearning_config = LoraUnlearningConfig.artbench_post_impressionism_config
        else:
            raise NotImplementedError
        args.lora_dir = unlearning_config["lora_dir"]
        args.lora_steps = None
        args.max_train_steps = unlearning_config["max_train_steps"]

    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.method)
        args.model_outdir = os.path.join(args.output_dir, "models", removal_dir)
        args.sample_outdir = os.path.join(args.output_dir, "samples", removal_dir)

        # If trained weights already exist, skip the script.
        lora_weight_path = os.path.join(
            args.model_outdir, "pytorch_lora_weights.safetensors"
        )
        if os.path.exists(lora_weight_path):
            print(
                f"Found trained LoRA weights at {lora_weight_path}. Process cancelled."
            )
            sys.exit(0)  # Exit without raising an error.

    prompt_list_dict = {"artbench": list(PromptConfig.artbench_config.values())}

    if args.validation_prompt is not None:
        if args.validation_prompt == "all":
            validation_prompt_list = prompt_list_dict[args.dataset]
        else:
            validation_prompt_list = [args.validation_prompt]

    logging_dir = Path(args.output_dir, args.logging_dir, removal_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.model_outdir, exist_ok=True)
            os.makedirs(args.sample_outdir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.model_outdir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    if args.lora_dir is None:
        unet.to(accelerator.device, dtype=weight_dtype)
        unet_lora_parameters = []
        for attn_processor_name, attn_processor in unet.attn_processors.items():
            # Parse the attention module.
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # Set the `lora_layer` attribute of the attention-related matrices.
            attn_module.to_q.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_q.in_features,
                    out_features=attn_module.to_q.out_features,
                    rank=args.rank,
                ).to(device=accelerator.device)
            )
            attn_module.to_k.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_k.in_features,
                    out_features=attn_module.to_k.out_features,
                    rank=args.rank,
                ).to(device=accelerator.device)
            )

            attn_module.to_v.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_v.in_features,
                    out_features=attn_module.to_v.out_features,
                    rank=args.rank,
                ).to(device=accelerator.device)
            )
            attn_module.to_out[0].set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_out[0].in_features,
                    out_features=attn_module.to_out[0].out_features,
                    rank=args.rank,
                ).to(device=accelerator.device)
            )

            # Accumulate the LoRA params to optimize.
            unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())
    else:
        # Runtime bugfix when LoRA ranks are different across attention to_q, to_k, 
        # to_v, and to_out.
        fix_get_processor(unet)
        weight_name = "pytorch_lora_weights"
        if args.lora_steps is not None:
            weight_name += f"_{args.lora_steps}"
        weight_name += ".safetensors"

        unet.load_attn_procs(args.lora_dir, weight_name=weight_name)
        lora_file = os.path.join(args.lora_dir, weight_name)
        logger.info(f"LoRA weights loaded from {lora_file}")

        # Convert non-LoRA parameters to the specified precision.
        unet_state_dict = unet.state_dict()
        for name, param in unet_state_dict.items():
            if "lora_layer" not in name:
                unet_state_dict[name] = unet_state_dict[name].to(weight_dtype)
        unet.load_state_dict(unet_state_dict, assign=True)
        unet.requires_grad_(False)
        unet.to(accelerator.device)

        # Add LoRA parameters to a list for the optimizer.
        unet_lora_parameters = []
        for name, param in unet.named_parameters():
            if "lora_layer" in name:
                param.requires_grad_(True)
                unet_lora_parameters.append(param)
        total_num_lora_params = sum(
            [param.numel() for param in unet_lora_parameters]
        )
        if expected_num_lora_params is not None:
            assert total_num_lora_params == expected_num_lora_params

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    if args.cls is not None and args.cls_key is not None:
        cls_idx = np.where(np.array(dataset["train"][args.cls_key]) == args.cls)[0]
        dataset["train"] = dataset["train"].select(cls_idx)
        if "artbench" in args.dataset:
            assert dataset["train"].num_rows == 5000

    if args.removal_dist is not None or args.removal_rank_file is not None:
        # Load csv file containing indexed removal units.
        if args.cls is None:
            removal_unit_file = os.path.join(
                args.train_data_dir, f"{args.removal_unit}s.csv"
            )
        else:
            removal_unit_file = os.path.join(
                args.train_data_dir, f"{args.cls}_{args.removal_unit}s.csv"
            )
        removal_unit_df = pd.read_csv(removal_unit_file)
        
        # Get remaining and removed indices.
        removal_idx_file = os.path.join(args.model_outdir, "removal_idx.csv")
        if os.path.exists(removal_idx_file):
            removal_idx_df = pd.read_csv(removal_idx_file)
            print(f"Removal index file loaded from {removal_idx_file}")
            remaining_idx = removal_idx_df["idx"][removal_idx_df["remaining"]].to_numpy()
            removal_idx = removal_idx_df["idx"][~removal_idx_df["remaining"]].to_numpy()
        elif args.removal_dist is not None:
            if args.removal_dist == "shapley":
                remaining_idx, removed_idx = remove_data_by_shapley(
                    removal_unit_df, args.removal_seed
                )
            elif args.removal_dist == "uniform":
                remaining_idx, removed_idx = remove_data_by_uniform(
                    removal_unit_df, args.removal_seed
                )
            elif args.removal_dist == "datamodel":
                remaining_idx, removed_idx = remove_data_by_datamodel(
                    dataset=removal_unit_df,
                    seed=args.removal_seed,
                    alpha=args.datamodel_alpha,
                )
            elif args.removal_dist == "loo":
                remaining_idx, removed_idx = remove_data_by_loo(
                    dataset=removal_unit_df,
                    loo_idx=args.loo_idx,
                )
            elif args.removal_dist == "aoi":
                remaining_idx, removed_idx = remove_data_for_aoi(
                    dataset=removal_unit_df,
                    aoi_idx=args.aoi_idx,
                )
            else:
                raise ValueError(
                    f"--removal_dist={args.removal_dist} has to be ['shapley', 'uniform', 'datamodel', 'loo', 'aoi']"
                )
            removal_idx_df = pd.concat(
                [
                    pd.DataFrame({"idx": remaining_idx, "remaining": True}),
                    pd.DataFrame({"idx": removed_idx, "remaining": False}),
                ]
            )
            removal_idx_df.to_csv(removal_idx_file, index=False)
            print(f"Removal index file saved to {removal_idx_file}")
        else:
            with open(args.removal_rank_file, "rb") as handle:
                removal_rank = np.load(handle)
            if args.removal_rank_proportion is not None:
                num_removed_units = math.floor(
                    len(removal_rank) * args.removal_rank_proportion
                )
                removed_idx = removal_rank[:num_removed_units]
                remaining_idx = removal_rank[num_removed_units:]
            else:
                num_removed_units = math.floor(
                    len(removal_rank) * args.removal_bottom_proportion
                )
                removed_idx = removal_rank[-num_removed_units:]
                remaining_idx = removal_rank[:-num_removed_units]

            removal_idx_df = pd.concat(
                [
                    pd.DataFrame({"idx": remaining_idx, "remaining": True}),
                    pd.DataFrame({"idx": removed_idx, "remaining": False}),
                ]
            )
            removal_idx_df.to_csv(removal_idx_file, index=False)
            print(f"Removal index file saved to {removal_idx_file}")
        
        # Remove data.
        kept_units = removal_unit_df.iloc[remaining_idx, 0].tolist()
        train_units = np.array(dataset["train"][args.removal_unit])
        dataset["train"] = dataset["train"].select(
            np.where(np.isin(train_units, kept_units))[0]
        )
        assert set(dataset["train"][args.removal_unit]) == set(kept_units)
        if args.removal_unit == "filename":
            assert dataset["train"].num_rows == len(remaining_idx)
    
    # If all data points are removed, save the LoRA weights and exit.
    if dataset["train"].num_rows == 0:
        unet.save_attn_procs(args.model_outdir)
        print(
            "All data points are removed. LoRA weights saved without further training."
        )
        sys.exit(0)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    image_column, caption_column = column_names[0], column_names[1]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.model_outdir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.model_outdir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # CSV file to record training time.
    time_file = os.path.join(args.model_outdir, "time.csv")
    if not os.path.exists(time_file) and accelerator.is_main_process:
        with open(time_file, "w") as f:
            if args.max_train_steps is None:
                f.write("epoch,time,gpu\n")
            else:
                f.write("step,time,gpu\n")
    for epoch in range(first_epoch, args.num_train_epochs):
        if args.max_train_steps is None and accelerator.is_main_process:
            epoch_start_time = time.time()
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.max_train_steps is not None and accelerator.is_main_process:
                step_start_time = time.time()
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.max_train_steps is not None and accelerator.is_main_process:
                    step_time = time.time() - step_start_time
                    time_record = f"{global_step},{step_time:.8f},{torch.cuda.get_device_name()}\n"
                    with open(time_file, "a") as f:
                        f.write(time_record)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.model_outdir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.model_outdir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.model_outdir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # Also save the LoRA weights for later usage.
                        # Unlike the training states, there are no limits to the number
                        # of LoRA weight files.
                        if args.checkpoint_attn_procs:
                            weight_name = f"pytorch_lora_weights_{global_step}.safetensors"
                            unet.save_attn_procs(
                                args.model_outdir,
                                weight_name=weight_name,
                            )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                args.validation_prompt is not None
                and epoch % args.validation_epochs == 0
            ):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=False),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline.safety_checker = None
                pipeline.requires_safety_checker = False
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                for prompt in validation_prompt_list:
                    for _ in tqdm(range(args.num_validation_images)):
                        images.append(
                            pipeline(
                                prompt,
                                num_inference_steps=100,
                                generator=generator,
                                height=args.resolution,
                                width=args.resolution,
                            ).images[0]
                        )
                if args.sample_outdir is not None:
                    torch_images = torch.stack([to_tensor(img) for img in images])
                    save_image(
                        torch_images,
                        os.path.join(args.sample_outdir, f"steps_{global_step}.png"),
                        nrow=args.num_validation_images,
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC"
                        )
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {args.validation_prompt}"
                                    )
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()
            if args.max_train_steps is None:
                epoch_time = time.time() - epoch_start_time
                time_record = f"{epoch},{epoch_time:.8f},{torch.cuda.get_device_name()}\n"
                with open(time_file, "a") as f:
                    f.write(time_record)
                    print(f"Epoch training time recorded at {time_file}")

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.model_outdir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.model_outdir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.model_outdir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    # Final inference
    # Load previous pipeline
    if accelerator.is_main_process and args.validation_prompt is not None:
        logger.info("Running final inference...")
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(
            args.model_outdir, weight_name="pytorch_lora_weights.safetensors"
        )

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed)
        images = []
        for prompt in validation_prompt_list:
            for _ in tqdm(range(args.num_validation_images)):
                images.append(
                    pipeline(
                        prompt,
                        num_inference_steps=100,
                        generator=generator,
                        height=args.resolution,
                        width=args.resolution,
                    ).images[0]
                )

        if args.sample_outdir is not None:
            torch_images = torch.stack([to_tensor(img) for img in images])
            save_image(
                torch_images,
                os.path.join(args.sample_outdir, f"steps_{global_step}.png"),
                nrow=args.num_validation_images,
            )

        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
    print("Done!")
