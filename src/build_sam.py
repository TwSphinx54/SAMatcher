# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import List, Optional  # Added for type hinting

HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    preserved_layers_prefixes: Optional[List[str]] = None,  # Changed parameter
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    
    # Pass the list of prefixes (or None) to _load_checkpoint
    _load_checkpoint(model, ckpt_path, preserve_layers=preserved_layers_prefixes) # Use the passed list
    
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_hq_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_hq_video_predictor.SAM2HQVideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path, preserve_layers: Optional[List[str]] = None):
    if not ckpt_path:
        logging.info("No checkpoint path provided. Model initialized with default weights.")
        return

    logging.info(f"Loading checkpoint from {ckpt_path}")
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    except Exception as e:
        logging.error(f"Error loading checkpoint file {ckpt_path}: {e}. Model will use initialized weights.")
        return

    if preserve_layers:
        logging.info(f"Attempting to selectively load layers with prefixes: {preserve_layers}")
        
        filtered_sd = {
            k: v for k, v in sd.items() 
            if any(k.startswith(prefix + ".") or k == prefix for prefix in preserve_layers)
        }
        
        if not filtered_sd:
            logging.warning(
                f"No weights in checkpoint '{ckpt_path}' matched the allowed prefixes: {preserve_layers}. "
                "Specified layers will use initialized weights."
            )
            # Even if filtered_sd is empty, call load_state_dict to get accurate missing_keys for preserved layers
            # This helps in reporting if model expected something for these layers.
            # Pass an empty dict if filtered_sd is truly empty.
            missing_keys, unexpected_keys = model.load_state_dict(filtered_sd or {}, strict=False)

        else:
            logging.info(f"Found {len(filtered_sd)} tensors matching specified prefixes for selective loading.")
            missing_keys, unexpected_keys = model.load_state_dict(filtered_sd, strict=False)

        loaded_count = len(filtered_sd) - len(unexpected_keys)
        
        if unexpected_keys:
            logging.warning(
                f"Unexpected keys found in checkpoint and not loaded: {unexpected_keys}."
            )

        if loaded_count > 0:
            logging.info(f"Successfully loaded {loaded_count} tensors for the specified layers.")
        elif filtered_sd: # filtered_sd was not empty, but all loaded keys were unexpected
            logging.warning("All tensors selected for loading were unexpected by the model structure.")
        
        # Identify keys that were expected for preserved layers but were not loaded
        missing_keys_for_preserved = [
            k for k in missing_keys 
            if any(k.startswith(p + ".") or k == p for p in preserve_layers)
        ]

        if missing_keys_for_preserved:
            logging.warning(
                f"Weights for these specified preserved layers were expected by model but not found in checkpoint or not loaded: "
                f"{sorted(list(set(missing_keys_for_preserved)))}. These parts will use initialized weights."
            )
        
        if loaded_count > 0 and not unexpected_keys and not missing_keys_for_preserved:
            logging.info("Selective checkpoint loading for specified layers completed successfully.")
        elif preserve_layers: # If an attempt for selective loading was made
            logging.info("Selective checkpoint loading completed with warnings/issues as noted above.")

    else: # Full checkpoint loading
        logging.info(f"Loading full checkpoint from {ckpt_path}.")
        try:
            model.load_state_dict(sd, strict=True)
            logging.info("Loaded full checkpoint successfully.")
        except RuntimeError as e:
            logging.error(f"Error loading full checkpoint (strict=True): {e}. Model weights may be incorrect.")
            # Potentially re-raise e if strict loading failure should halt execution
