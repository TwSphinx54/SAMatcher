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
    layer_prefixes_to_freeze = None

    if preserved_layers_file_path:
        try:
            with open(preserved_layers_file_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'layer_config' in config:
                    layer_config_data = config['layer_config']
                    layer_prefixes_to_load = layer_config_data.get('load_weights', [])
                    layer_prefixes_to_freeze = layer_config_data.get('freeze_layers', [])
                else:
                    layer_prefixes_to_load = []
                    layer_prefixes_to_freeze = []

            if not layer_prefixes_to_load and not layer_prefixes_to_freeze:
                logging.info(
                    f"Preserved layers file '{preserved_layers_file_path}' is empty. "
                    "No layers will be specifically loaded or frozen based on this file."
                )
                layer_prefixes_to_load = None
                layer_prefixes_to_freeze = None
            else:
                logging.info(
                    f"Read {len(layer_prefixes_to_load)} load layer prefixes and "
                    f"{len(layer_prefixes_to_freeze)} freeze layer prefixes from '{preserved_layers_file_path}'."
                )
        except FileNotFoundError:
            logging.warning(
                f"Preserved layers file not found: {preserved_layers_file_path}. "
                "Proceeding without specific layer loading or freezing."
            )
            layer_prefixes_to_load = None
            layer_prefixes_to_freeze = None
        except Exception as e:
            logging.error(
                f"Error reading preserved layers file {preserved_layers_file_path}: {e}. "
                "Proceeding without specific layer loading or freezing."
            )
            layer_prefixes_to_load = None
            layer_prefixes_to_freeze = None

    model = build_sam2(
        config_file=config_file,
        ckpt_path=ckpt_path,
        preserved_layers_prefixes=layer_prefixes_to_load,
        mode="train"
    )

    if layer_prefixes_to_freeze:
        logging.info(f"Attempting to freeze layers with prefixes: {layer_prefixes_to_freeze}")
        frozen_params_count = 0
        total_params_in_frozen_layers = 0
        
        for name, param in model.named_parameters():
            is_in_frozen_layer = False
            for prefix in layer_prefixes_to_freeze:
                if name.startswith(prefix + ".") or name == prefix:
                    is_in_frozen_layer = True
                    break
            
            if is_in_frozen_layer:
                total_params_in_frozen_layers += 1
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params_count += 1

        if total_params_in_frozen_layers > 0:
            logging.info(
                f"Froze {frozen_params_count} parameters out of {total_params_in_frozen_layers} "
                f"parameters belonging to layers with specified prefixes: {layer_prefixes_to_freeze}."
            )
            if frozen_params_count < total_params_in_frozen_layers:
                logging.info(
                    f"{total_params_in_frozen_layers - frozen_params_count} parameters in the specified layers "
                    "were already frozen or did not exist."
                )
        else:
            logging.info(
                f"No parameters found matching the prefixes {layer_prefixes_to_freeze} to freeze."
            )
    elif preserved_layers_file_path:
        logging.info("No valid layer prefixes found for freezing after processing the preserved layers file.")
    else:
        logging.info("No preserved_layers_file_path provided. All trainable layers will remain trainable.")

    return model