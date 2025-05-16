import math
import torch
import pprint
import argparse
import warnings
import time
from pathlib import Path
from src.build_model import ModelTrainer
from src.build_samatcher import build_samatcher
from src.data import MultiSceneDataModule
from loguru import logger as loguru_logger
from configs.default import get_cfg_defaults
from accelerate import Accelerator
from accelerate.utils import set_seed
from src.utils.profiler import build_profiler
from src.utils.optimizers import build_optimizer, build_scheduler

warnings.filterwarnings("ignore") # Suppress warnings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_cfg_path', type=str, default='config/train_config.py', help='Path to data configuration file.')
    parser.add_argument('--exp_name', type=str, default='SAMatcher', help='Experiment name, used for logging and output directories.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save checkpoints and logs.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to a pretrained checkpoint to load from for resuming training or fine-tuning.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l_sammatcher.yaml", help="SAM config.")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam2.1_hq_hiera_large.pt", help="The path to the SAM checkpoint to use for mask generation.")

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU. The effective batch size will be (num_gpus * batch_size * gradient_accumulation_steps).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to load data to pinned memory for faster CPU to GPU transfers.')
    parser.add_argument('--true_lr', type=float, default=1e-4, help='Base learning rate. This will be used directly or scaled if canonical_bs is set in config.')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating model weights. Simulates a larger batch size.')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help="Enable mixed precision training ('no', 'fp16', 'bf16') for speed and memory efficiency.")

    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log training metrics every N optimizer steps.')
    parser.add_argument('--val_every_n_epochs', type=int, default=1, help='Run validation every N epochs.')
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--wandb_entity', type=str, default=None, help="Weights & Biases entity name (username or team name).")
    parser.add_argument('--wandb_project', type=str, default=None, help="Weights & Biases project name (overrides exp_name for W&B).")
    
    parser.add_argument('--profiler_name', type=str, default=None, help='Profiler to use (e.g., "pytorch", "inference"). Leave None to disable.')

    # Add val_batch_size to args if not present, for MultiSceneDataModule
    parser.add_argument('--val_batch_size', type=int, default=2, help='Batch size for validation and testing per GPU.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Initialize Accelerator and set seed ---
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["wandb"],
        project_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    # --- Setup Logging ---
    # Configure loguru to use accelerator.print for distributed-friendly logging.
    loguru_logger.remove() 
    loguru_logger.add(lambda msg: accelerator.print(msg), format="{time} {level} {message}", level="INFO")
    
    # --- Initialize Weights & Biases (on main process only) ---
    if accelerator.is_main_process:
        wandb_project_name = args.wandb_project if args.wandb_project else args.exp_name
        current_time_str = time.strftime("%Y%m%d-%H%M%S")
        custom_run_name = f"{args.exp_name}_{current_time_str}" 
        
        init_kwargs = {"wandb": {"name": custom_run_name}}
        if args.wandb_entity:
            init_kwargs["wandb"]["entity"] = args.wandb_entity
        
        accelerator.init_trackers(project_name=wandb_project_name, config=vars(args), init_kwargs=init_kwargs)
        loguru_logger.info(f"Run arguments: {pprint.pformat(vars(args))}")

    # --- Load and adapt configuration ---
    config = get_cfg_defaults()
    config.merge_from_file(args.data_cfg_path)
    
    # Adapt config based on Accelerate's distributed setup and runtime arguments
    config.TRAINER.WORLD_SIZE = accelerator.num_processes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size * args.gradient_accumulation_steps
    config.TRAINER.TRUE_LR = args.true_lr
    
    # Scaling logic is removed.
    # Initialize WARMUP_STEP.
    # If WARMUP_EPOCHS is set (>0) in config, WARMUP_STEP will be calculated after dataloader initialization.
    # Otherwise, WARMUP_STEP from config (e.g., default.py or train_config.py) will be used.
    if not (hasattr(config.TRAINER, 'WARMUP_STEP') and config.TRAINER.WARMUP_STEP > 0):
        config.TRAINER.WARMUP_STEP = 0 

    K_BEST_CHECKPOINTS = 5
    top_k_checkpoints = [] 

    # Pass accelerator to allow DataModule to leverage its properties (e.g., for distributed sampling).
    data_module = MultiSceneDataModule(args, config, accelerator) 
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader() # Can be a single DataLoader or a list of DataLoaders

    # --- Calculate Warmup Optimizer Steps from WARMUP_EPOCHS ---
    if hasattr(config.TRAINER, 'WARMUP_EPOCHS') and config.TRAINER.WARMUP_EPOCHS > 0:
        if train_dataloader is not None and len(train_dataloader) > 0:
            steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            
            if steps_per_epoch == 0:
                if accelerator.is_main_process:
                    loguru_logger.warning(
                        f"Calculated 0 optimizer steps_per_epoch for warmup "
                        f"(len(train_dataloader)={len(train_dataloader)}, "
                        f"grad_accum_steps={args.gradient_accumulation_steps}). "
                        f"This can happen if the dataset size per process is smaller than gradient_accumulation_steps. "
                        f"Setting WARMUP_STEP to 0."
                    )
                config.TRAINER.WARMUP_STEP = 0 
            else:
                config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_EPOCHS * steps_per_epoch)
        else: 
            if accelerator.is_main_process:
                loguru_logger.warning(
                    "Train dataloader is None or empty. Cannot calculate WARMUP_STEP from WARMUP_EPOCHS. "
                    "Setting WARMUP_STEP to 0."
                )
            config.TRAINER.WARMUP_STEP = 0
    # If WARMUP_EPOCHS was not used (or <=0), config.TRAINER.WARMUP_STEP retains its value from the config files.
    if config.TRAINER.WARMUP_STEP < 0: # Ensure WARMUP_STEP is non-negative.
        config.TRAINER.WARMUP_STEP = 0

    if accelerator.is_main_process:
        loguru_logger.info(f"World size (num_processes): {config.TRAINER.WORLD_SIZE}")
        loguru_logger.info(f"Effective batch size (world_size * batch_size_per_gpu * grad_accum_steps): {config.TRAINER.TRUE_BATCH_SIZE}")
        loguru_logger.info(f"Base learning rate: {config.TRAINER.TRUE_LR}")
        loguru_logger.info(f"Final warmup optimizer steps: {config.TRAINER.WARMUP_STEP}")

    # --- Initialize Model, Optimizer, and Scheduler ---
    profiler_output_path = Path(args.output_dir) / args.exp_name / "profiler"
    profiler = build_profiler(args.profiler_name, accelerator, output_dir=profiler_output_path)
    
    # ModelTrainer is a handler class that encapsulates the model and training/validation logic
    sam_model = build_samatcher(args.sam_cfg, args.sam_checkpoint)
    model_handler = ModelTrainer(config, sam_model=sam_model, pretrained_ckpt=args.ckpt_path, profiler=profiler, accelerator=accelerator)
    model = model_handler.model

    # This section creates parameter groups for the optimizer, allowing different LRs for different parts of the model.
    # E.g., 'mask_decoder' parameters might use a smaller LR for fine-tuning.
    finetune_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'mask_decoder' in name: 
                finetune_params.append(param)
            else:
                new_params.append(param)
    
    param_groups = []
    if finetune_params:
        param_groups.append({
            'params': finetune_params, 
            'lr': config.TRAINER.TRUE_LR * config.TRAINER.FT_LR_SCALE 
        })
    if new_params:
        param_groups.append({
            'params': new_params, 
            'lr': config.TRAINER.TRUE_LR
        })

    # If no specific parameter groups are defined, use all trainable parameters for the optimizer. Otherwise, use the defined groups.
    optimizer_params = param_groups if param_groups else filter(lambda p: p.requires_grad, model.parameters())
        
    optimizer = build_optimizer(optimizer_params, config)
    scheduler = build_scheduler(config, optimizer) # Scheduler can be None if not configured

    # --- Prepare components with Accelerator ---
    # This step wraps model, optimizer, dataloaders, and scheduler for distributed training, device placement, and mixed precision.
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # --- Training Loop ---
    total_steps = 0
    loguru_logger.info(f"Starting training for {args.num_epochs} epochs.")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        if train_dataloader is None: # Ensure train_dataloader is available
            loguru_logger.error("Train dataloader is None. Skipping training epoch.")
            break 

        for step, batch in enumerate(train_dataloader):
            if batch is None: # Handle cases where collate_fn_skip_none might return None
                loguru_logger.warning(f"Skipping step {step} in epoch {epoch} due to empty batch after filtering.")
                continue
            
            # Learning rate warmup phase (applied per optimizer step)
            if config.TRAINER.WARMUP_STEP > 0 and total_steps < config.TRAINER.WARMUP_STEP:
                if config.TRAINER.WARMUP_TYPE == 'linear':
                    base_lr = config.TRAINER.WARMUP_RATIO * config.TRAINER.TRUE_LR
                    current_lr_scale = total_steps / config.TRAINER.WARMUP_STEP
                    # For each param group, scale its target LR by current_lr_scale during warmup
                    for pg_idx, pg in enumerate(optimizer.param_groups):
                        # The 'lr' in param_groups is the target LR for that group
                        target_lr_for_group = param_groups[pg_idx]['lr'] if param_groups else config.TRAINER.TRUE_LR
                        pg['lr'] = base_lr + current_lr_scale * abs(target_lr_for_group - base_lr)
                # Add other warmup types (e.g., 'cosine') if necessary
            
            # Forward and backward pass with gradient accumulation
            # accelerator.accumulate handles gradient synchronization and accumulation.
            with accelerator.accumulate(model):
                # _trainval_inference performs forward pass, computes loss, and updates 'batch'
                model_handler._trainval_inference(batch) 
                loss = batch['loss'] 
                loss_scalars = batch['loss_scalars']

                accelerator.backward(loss)

                # Gradient clipping and optimizer step occur only when gradients are synchronized
                if accelerator.sync_gradients:
                    if config.TRAINER.GRADIENT_CLIPPING > 0:
                        accelerator.clip_grad_norm_(model.parameters(), config.TRAINER.GRADIENT_CLIPPING)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1 # Increment total_steps only when optimizer.step() is called

            # Step-wise learning rate scheduler (if configured and gradients were synchronized)
            if scheduler is not None and scheduler['interval'] == 'step' and accelerator.sync_gradients:
                 unwrapped_scheduler = accelerator.unwrap_model(scheduler['scheduler'])
                 unwrapped_scheduler.step()

            # Log training metrics (on main process only)
            if total_steps > 0 and total_steps % args.log_every_n_steps == 0 and accelerator.is_main_process:
                log_dict = {"epoch": epoch, "step": total_steps, "train_loss": loss.item()}
                for k, v in loss_scalars.items():
                    log_dict[f"train_{k}"] = v.item() if hasattr(v, 'item') else v
                
                # Log learning rates for all parameter groups
                if optimizer and hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                    for i, pg in enumerate(optimizer.param_groups):
                        log_dict[f"lr_group_{i}"] = pg['lr']
                accelerator.log(log_dict, step=total_steps)

                # Calculate and print ETA for the current epoch
                steps_in_epoch = len(train_dataloader)
                time_per_step_epoch = (time.time() - epoch_start_time) / (step + 1)
                remaining_steps_epoch = steps_in_epoch - (step + 1)
                eta_epoch_seconds = remaining_steps_epoch * time_per_step_epoch
                eta_epoch_str = time.strftime("%H:%M:%S", time.gmtime(eta_epoch_seconds))
                
                total_optimizer_steps_all_epochs = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_epochs
                
                # Prepare learning rate string for console logging
                lr_info_parts = []
                if optimizer and hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                    for i, pg in enumerate(optimizer.param_groups):
                        lr_info_parts.append(f"LR_g{i}: {pg['lr']:.2e}")
                lr_info_str = ", ".join(lr_info_parts) if lr_info_parts else "LR: N/A"
                
                loguru_logger.info(f"Epoch {epoch}, Step {total_steps}/{total_optimizer_steps_all_epochs}, Batch {step+1}/{steps_in_epoch}, ETA: {eta_epoch_str}, Loss: {loss.item():.4f}, {lr_info_str}")

        # End of epoch scheduler step (if configured)
        if scheduler is not None and scheduler['interval'] == 'epoch':
            unwrapped_scheduler = accelerator.unwrap_model(scheduler['scheduler'])
            unwrapped_scheduler.step()

        # --- Validation Phase ---
        if epoch % args.val_every_n_epochs == 0:
            model.eval()
            all_val_loss_scalars_batches = [] 
            all_val_metrics_batches = []      
            
            loguru_logger.info(f"Running validation for epoch {epoch}...")
            
            if val_dataloader is None:
                loguru_logger.warning("Val dataloader is None. Skipping validation phase.")
            else:
                current_val_loaders = val_dataloader if isinstance(val_dataloader, list) else [val_dataloader]

                for val_loader_idx, current_val_loader in enumerate(current_val_loaders):
                    if current_val_loader is None:
                        loguru_logger.warning(f"Validation dataloader at index {val_loader_idx} is None. Skipping.")
                        continue
                    
                    dataset_name_prefix = f"val_ds{val_loader_idx}_" if len(current_val_loaders) > 1 else "val_"

                    for val_step, val_batch in enumerate(current_val_loader):
                        if val_batch is None: 
                            loguru_logger.warning(f"Skipping val_step {val_step} in val_loader {val_loader_idx} due to empty batch.")
                            continue
                        with torch.no_grad():
                            model_handler._trainval_inference(val_batch) 
                            metrics_output = model_handler._compute_metrics(val_batch) 
                        
                        # The following explicit .to(accelerator.device) calls are defensive.
                        # If _trainval_inference and _compute_metrics always return tensors on accelerator.device, these might be redundant.
                        # accelerator.gather_for_metrics should handle device placement.
                        if 'loss_scalars' in val_batch and val_batch['loss_scalars'] is not None:
                            processed_loss_scalars = {}
                            source_data = val_batch['loss_scalars']
                            if isinstance(source_data, dict):
                                for key, tensor_val in source_data.items():
                                    if not isinstance(tensor_val, torch.Tensor): # Handle non-tensor values
                                        tensor_val = torch.tensor(tensor_val, device=accelerator.device, dtype=torch.float32)
                                    processed_loss_scalars[key] = tensor_val.to(accelerator.device)
                            elif isinstance(source_data, torch.Tensor): # Fallback for unexpected format
                                processed_loss_scalars = {'loss': source_data.to(accelerator.device)}
                            val_batch['loss_scalars'] = processed_loss_scalars if processed_loss_scalars else {}


                        if 'metrics' in metrics_output and metrics_output['metrics'] is not None:
                            processed_metrics = {}
                            source_data = metrics_output['metrics']
                            if isinstance(source_data, dict):
                                for key, tensor_val in source_data.items():
                                    if not isinstance(tensor_val, torch.Tensor): # Handle non-tensor values
                                        tensor_val = torch.tensor(tensor_val, device=accelerator.device, dtype=torch.float32)
                                    processed_metrics[key] = tensor_val.to(accelerator.device)
                            elif isinstance(source_data, torch.Tensor): # Fallback for unexpected format
                                processed_metrics = {'metric': source_data.to(accelerator.device)}
                            metrics_output['metrics'] = processed_metrics if processed_metrics else {}

                        # Gather loss scalars and metrics from all processes.
                        # `gather_for_metrics` is used as these are tensors intended for metric computation/logging.
                        gathered_loss_scalars = accelerator.gather_for_metrics(val_batch.get('loss_scalars', {}))
                        gathered_metrics = accelerator.gather_for_metrics(metrics_output.get('metrics', {}))
                        
                        prefixed_gathered_loss_scalars = {f"{dataset_name_prefix}{k}": v for k, v in gathered_loss_scalars.items()}
                        prefixed_gathered_metrics = {f"{dataset_name_prefix}metric_{k}": v for k, v in gathered_metrics.items()}

                        if prefixed_gathered_loss_scalars: # Only append if there's data
                            all_val_loss_scalars_batches.append(prefixed_gathered_loss_scalars)
                        if prefixed_gathered_metrics: # Only append if there's data
                            all_val_metrics_batches.append(prefixed_gathered_metrics)

                        if accelerator.is_main_process and val_step < model_handler.n_vals_plot:
                            vis_prefix = f"{dataset_name_prefix}epoch{epoch}"
                            model_handler._visualize_batch(val_batch, val_batch, val_step, prefix=vis_prefix, global_step=total_steps)

            # Aggregate and log validation metrics (on main process only)
            if accelerator.is_main_process:
                avg_val_loss_scalars = {}
                if all_val_loss_scalars_batches:
                    all_keys = set(k for d in all_val_loss_scalars_batches for k in d.keys())
                    for key in all_keys:
                        # d[key] is a gathered tensor (from multiple devices for that batch)
                        # Concatenate these tensors from all batches, then take the mean.
                        valid_tensors = [d[key] for d in all_val_loss_scalars_batches if key in d and isinstance(d[key], torch.Tensor) and d[key].numel() > 0]
                        if valid_tensors:
                             avg_val_loss_scalars[key] = torch.cat(valid_tensors).float().mean().item()

                avg_val_metrics = {}
                if all_val_metrics_batches:
                    all_keys = set(k for d in all_val_metrics_batches for k in d.keys())
                    for key in all_keys:
                        valid_tensors = [d[key] for d in all_val_metrics_batches if key in d and isinstance(d[key], torch.Tensor) and d[key].numel() > 0]
                        if valid_tensors:
                            avg_val_metrics[key] = torch.cat(valid_tensors).float().mean().item()
                
                log_val_dict = {**avg_val_loss_scalars, **avg_val_metrics}
                if log_val_dict: 
                    accelerator.log(log_val_dict, step=total_steps)
                    accelerator.print(f"Validation Epoch {epoch} Results: {pprint.pformat(log_val_dict)}")

                    # Top-K checkpoint saving logic
                    current_epoch_metric_for_top_k = -float('inf')
                    # Using the average of all reported validation metrics for ranking
                    selected_metric_name_for_top_k = "average_of_all_val_metrics" 
                    
                    if avg_val_metrics:
                        metric_values = [v for v in avg_val_metrics.values() if isinstance(v, (int, float))]
                        if metric_values:
                            current_epoch_metric_for_top_k = sum(metric_values) / len(metric_values)
                            accelerator.print(f"Using metric '{selected_metric_name_for_top_k}' for top-K ranking: {current_epoch_metric_for_top_k:.4f}")
                        else:
                            accelerator.print("No numerical metric values found in avg_val_metrics to calculate an average for top-K ranking.")
                    else:
                        accelerator.print("avg_val_metrics is empty. Cannot determine metric for top-K checkpoint.")

                    if current_epoch_metric_for_top_k > -float('inf'): # Proceed if a valid metric was calculated
                        save_dir = Path(args.output_dir) / args.exp_name
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        if len(top_k_checkpoints) < K_BEST_CHECKPOINTS or current_epoch_metric_for_top_k > top_k_checkpoints[-1][0]:
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_ckpt_filename = f"best_avg_metric_{current_epoch_metric_for_top_k:.4f}_epoch_{epoch}_step_{total_steps}.ckpt"
                            best_ckpt_save_path = save_dir / best_ckpt_filename
                            
                            accelerator.save(unwrapped_model.state_dict(), str(best_ckpt_save_path))
                            accelerator.print(f"Saved new top-{K_BEST_CHECKPOINTS} checkpoint: {best_ckpt_save_path} with {selected_metric_name_for_top_k}: {current_epoch_metric_for_top_k:.4f}")
                            
                            top_k_checkpoints.append((current_epoch_metric_for_top_k, str(best_ckpt_save_path)))
                            top_k_checkpoints.sort(key=lambda x: x[0], reverse=True) 
                            
                            if len(top_k_checkpoints) > K_BEST_CHECKPOINTS:
                                removed_ckpt_info = top_k_checkpoints.pop() 
                                removed_ckpt_path_str = removed_ckpt_info[1]
                                try:
                                    Path(removed_ckpt_path_str).unlink(missing_ok=True)
                                    accelerator.print(f"Removed old top-{K_BEST_CHECKPOINTS} checkpoint: {removed_ckpt_path_str}")
                                except OSError as e:
                                    accelerator.print(f"Error removing old top-{K_BEST_CHECKPOINTS} checkpoint {removed_ckpt_path_str}: {e}")
                            
                            accelerator.print(f"Current top-{K_BEST_CHECKPOINTS} checkpoints (Metric: {selected_metric_name_for_top_k}):")
                            for score, path_str in top_k_checkpoints:
                                accelerator.print(f"  Score: {score:.4f}, Path: {path_str}")
                else:
                    accelerator.print(f"Validation Epoch {epoch}: No metrics to log.")

        # --- Save Checkpoint (on main process only) ---
        if epoch % args.save_every_n_epochs == 0 and accelerator.is_main_process:
            save_dir = Path(args.output_dir) / args.exp_name
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_filename = f"epoch_{epoch}_step_{total_steps}.ckpt"
            save_path = save_dir / ckpt_filename
            
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), str(save_path))
            accelerator.print(f"Saved checkpoint (model weights) to {save_path}")

    # --- End of Training ---
    if accelerator.is_main_process:
        if profiler:
             profiler.summary()
        accelerator.end_training()
        loguru_logger.info("Training finished.")

if __name__ == "__main__":
    main()
