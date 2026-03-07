# filepath: /data/nfs/px/Projects/SAMatcher/src/build_samatcher.py
import logging
import yaml
from src.build_sam import build_sam2
from src.modeling.sam2_base import SAM2Base


def build_samatcher(
    config_file='configs/sam2.1/sam2.1_hq_hiera_l.yaml',
    ckpt_path='checkpoints/sam2.1_hq_hiera_large.pt',
    preserved_layers_file_path='configs/sam_2.1_hiera_preserved_args.yaml'
) -> SAM2Base:

    layer_prefixes_to_load = None
    layer_prefixes_to_train = None # Changed from layer_prefixes_to_freeze

    if preserved_layers_file_path:
        try:
            with open(preserved_layers_file_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'layer_config' in config:
                    layer_config_data = config['layer_config']
                    layer_prefixes_to_load = layer_config_data.get('load_weights', [])
                    # Read 'train_layers' instead of 'freeze_layers'
                    layer_prefixes_to_train = layer_config_data.get('train_layers', [])
                else:
                    layer_prefixes_to_load = []
                    layer_prefixes_to_train = [] # Changed from layer_prefixes_to_freeze

            if not layer_prefixes_to_load and not layer_prefixes_to_train: # Adjusted condition
                logging.info(
                    f"Preserved layers file '{preserved_layers_file_path}' is empty or does not specify layers to train. "
                    "No layers will be specifically loaded or trained based on this file."
                )
                layer_prefixes_to_load = None
                layer_prefixes_to_train = None # Changed from layer_prefixes_to_freeze
            else:
                logging.info(
                    f"Read {len(layer_prefixes_to_load if layer_prefixes_to_load else [])} load layer prefixes and " # Adjusted for potential None
                    f"{len(layer_prefixes_to_train if layer_prefixes_to_train else [])} train layer prefixes from '{preserved_layers_file_path}'." # Adjusted for potential None
                )
        except FileNotFoundError:
            logging.warning(
                f"Preserved layers file not found: {preserved_layers_file_path}. "
                "Proceeding without specific layer loading or training."
            )
            layer_prefixes_to_load = None
            layer_prefixes_to_train = None # Changed from layer_prefixes_to_freeze
        except Exception as e:
            logging.error(
                f"Error reading preserved layers file {preserved_layers_file_path}: {e}. "
                "Proceeding without specific layer loading or training."
            )
            layer_prefixes_to_load = None
            layer_prefixes_to_train = None # Changed from layer_prefixes_to_freeze

    model = build_sam2(
        config_file=config_file,
        ckpt_path=ckpt_path,
        preserved_layers_prefixes=layer_prefixes_to_load,
        mode="train"
    )

    # Freeze all parameters first
    logging.info("Freezing all model parameters initially.")
    for param in model.parameters():
        param.requires_grad = False

    total_params_count = sum(p.numel() for p in model.parameters())
    trainable_params_count = 0

    if layer_prefixes_to_train:
        logging.info(f"Attempting to unfreeze layers for training with prefixes: {layer_prefixes_to_train}")
        unfrozen_params_count = 0

        for name, param in model.named_parameters():
            is_in_train_layer = False
            for prefix in layer_prefixes_to_train:
                if name.startswith(prefix + ".") or name == prefix:
                    is_in_train_layer = True
                    break

            if is_in_train_layer:
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # count individual parameters
                trainable_params_count += param.numel() # count all params in trainable layers
            elif param.requires_grad: # This case should ideally not happen if all were frozen
                logging.warning(f"Parameter {name} was expected to be frozen but requires_grad is True. Freezing it now.")
                param.requires_grad = False


        if unfrozen_params_count > 0:
            logging.info(
                f"Unfroze {unfrozen_params_count} parameters belonging to layers with specified prefixes: {layer_prefixes_to_train}."
            )
        else:
            logging.info(
                f"No parameters were unfrozen. Ensure 'train_layers' in '{preserved_layers_file_path}' "
                f"are correctly specified and exist in the model. Or, they might have been already trainable (which shouldn't happen with initial freeze)."
            )

        # Update trainable_params_count to reflect only newly unfrozen params if that's desired,
        # or sum up all params that are now trainable.
        # Current logic sums all params in layers marked for training.
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters in the model: {total_params_count}")
        logging.info(f"Total trainable parameters after selective unfreezing: {final_trainable_params}")

    elif preserved_layers_file_path: # This means layer_prefixes_to_train was empty or None
        logging.info(
            f"No 'train_layers' specified in '{preserved_layers_file_path}' or the list was empty. "
            "All model parameters will remain frozen as per initial step."
        )
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters in the model: {total_params_count}")
        logging.info(f"Total trainable parameters: {final_trainable_params} (should be 0 if no train_layers).")
    else: # No preserved_layers_file_path provided
        logging.info(
            "No preserved_layers_file_path provided. All model parameters will remain frozen as per initial step. "
            "This means the model will not train unless 'train_layers' are specified elsewhere or this logic is changed."
        )
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters in the model: {total_params_count}")
        logging.info(f"Total trainable parameters: {final_trainable_params} (should be 0).")


    return model
