import math
import torch
import pprint
import argparse
import warnings
import time
from pathlib import Path
from src.build_model import ModelTrainer
from src.data import MultiSceneDataModule
from loguru import logger as loguru_logger
from config.default import get_cfg_defaults
from accelerate import Accelerator
from accelerate.utils import set_seed
from src.utils.profiler import build_profiler
from src.utils.optimizers import build_optimizer, build_scheduler

warnings.filterwarnings("ignore") # Suppress warnings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General arguments
    parser.add_argument('--data_cfg_path', type=str, default='config/train_config.py', help='Path to data configuration file.')
    parser.add_argument('--exp_name', type=str, default='SegMatcher_Accelerate', help='Experiment name, used for logging and output directories.')
    parser.add_argument('--output_dir', type=str, default='outputs_accelerate', help='Directory to save checkpoints and logs.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to a pretrained checkpoint to load from for resuming training or fine-tuning.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    # Training specific arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU. The effective batch size will be (num_gpus * batch_size * gradient_accumulation_steps).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to load data to pinned memory for faster CPU to GPU transfers.')
    parser.add_argument('--true_lr', type=float, default=0.0001, help='Base learning rate. This will be used directly or scaled if canonical_bs is set in config.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating model weights. Simulates a larger batch size.')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help="Enable mixed precision training ('no', 'fp16', 'bf16') for speed and memory efficiency.")

    # Logging and saving arguments
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log training metrics every N optimizer steps.')
    parser.add_argument('--val_every_n_epochs', type=int, default=1, help='Run validation every N epochs.')
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--wandb_entity', type=str, default=None, help="Weights & Biases entity name (username or team name).")
    parser.add_argument('--wandb_project', type=str, default=None, help="Weights & Biases project name (overrides exp_name for W&B).")
    
    # Profiling
    parser.add_argument('--profiler_name', type=str, default=None, help='Profiler to use (e.g., "pytorch", "inference"). Leave None to disable.')

    # Add val_batch_size to args if not present, for MultiSceneDataModule
    parser.add_argument('--val_batch_size', type=int, default=2, help='Batch size for validation and testing per GPU.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Initialize Accelerator and set seed ---
    # Accelerator handles device placement, distributed training, mixed precision, etc.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["wandb"],  # Integrate with Weights & Biases for logging
        project_dir=args.output_dir, # Directory for Accelerator to store its own logs/outputs
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed) # Ensure reproducibility

    # --- Setup Logging ---
    # Configure loguru to use accelerator.print for distributed-friendly logging.
    # This ensures logs are printed only on the main process or as configured by Accelerator.
    loguru_logger.remove() 
    loguru_logger.add(lambda msg: accelerator.print(msg), format="{time} {level} {message}", level="INFO")
    
    # --- Initialize Weights & Biases (on main process only) ---
    if accelerator.is_main_process:
        wandb_project_name = args.wandb_project if args.wandb_project else args.exp_name
        
        current_time_str = time.strftime("%Y%m%d-%H%M%S")
        custom_run_name = f"{args.exp_name}_{current_time_str}" 
        
        init_kwargs = {"wandb": {}} # Pass wandb specific init args here
        if args.wandb_entity:
            init_kwargs["wandb"]["entity"] = args.wandb_entity
        
        init_kwargs["wandb"]["name"] = custom_run_name
        
        # Initialize trackers (like W&B)
        accelerator.init_trackers(project_name=wandb_project_name, config=vars(args), init_kwargs=init_kwargs)
        loguru_logger.info(f"Run arguments: {pprint.pformat(vars(args))}")

    # --- Load and adapt configuration ---
    config = get_cfg_defaults() # Load base configuration
    config.merge_from_file(args.data_cfg_path) # Merge with data-specific configuration
    
    # Adapt config based on Accelerate's distributed setup and runtime arguments
    config.TRAINER.WORLD_SIZE = accelerator.num_processes # Total number of GPUs/processes
    # Calculate true batch size considering number of GPUs and gradient accumulation
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size * args.gradient_accumulation_steps
    
    # Calculate scaling factor for learning rate and warmup steps based on a canonical batch size (if defined in config)
    # This helps maintain training stability when changing batch sizes.
    if config.TRAINER.TRUE_BATCH_SIZE > 0 and hasattr(config.TRAINER, 'CANONICAL_BS') and config.TRAINER.CANONICAL_BS > 0:
        _scaling = config.TRAINER.CANONICAL_BS / config.TRAINER.TRUE_BATCH_SIZE
    else:
        _scaling = 1.0
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = args.true_lr # Set true learning rate from args (could be scaled if needed)
    
    # Adjust warmup steps based on scaling, if WARMUP_STEP is defined and positive
    if hasattr(config.TRAINER, 'WARMUP_STEP') and config.TRAINER.WARMUP_STEP and config.TRAINER.WARMUP_STEP > 0:
        config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP * _scaling)
    else:
        config.TRAINER.WARMUP_STEP = 0

    # Initialize Top-K checkpoint tracking
    K_BEST_CHECKPOINTS = 5
    top_k_checkpoints = [] # List of tuples: (metric_score, path_str)

    if accelerator.is_main_process:
        loguru_logger.info(f"World size (num_processes): {config.TRAINER.WORLD_SIZE}")
        loguru_logger.info(f"Effective batch size (world_size * batch_size_per_gpu * grad_accum_steps): {config.TRAINER.TRUE_BATCH_SIZE}")
        loguru_logger.info(f"Base learning rate: {config.TRAINER.TRUE_LR}")
        loguru_logger.info(f"Adjusted warmup steps: {config.TRAINER.WARMUP_STEP}")

    # --- Prepare Dataloaders ---
    # MultiSceneDataModule handles dataset and dataloader creation.
    # Pass accelerator to allow DataModule to leverage its properties (e.g., for distributed sampling).
    data_module = MultiSceneDataModule(args, config, accelerator) 
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader() # Can be a single DataLoader or a list of DataLoaders
    
    # --- Initialize Model, Optimizer, and Scheduler ---
    # Setup profiler if specified
    profiler_output_path = Path(args.output_dir) / args.exp_name / "profiler"
    profiler = build_profiler(args.profiler_name, accelerator, output_dir=profiler_output_path)
    
    # ModelTrainer is a handler class that encapsulates the model and training/validation logic
    model_handler = ModelTrainer(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, accelerator=accelerator)
    model = model_handler.model # Get the actual torch.nn.Module

    # Optimizer and Scheduler setup
    # This section creates parameter groups for the optimizer, allowing different LRs for different parts of the model.
    # E.g., 'mask_decoder' parameters might use a smaller LR for fine-tuning.
    finetune_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Example: parameters containing 'mask_decoder' in their name are treated as fine-tuning parameters
            if 'mask_decoder' in name: 
                finetune_params.append(param)
            else:
                new_params.append(param)
    
    param_groups = []
    if finetune_params:
        param_groups.append({
            'params': finetune_params, 
            'lr': config.TRAINER.TRUE_LR * config.TRAINER.FT_LR_SCALE # Fine-tuning LR (e.g., 0.1 * base_lr)
        })
    if new_params:
        param_groups.append({
            'params': new_params, 
            'lr': config.TRAINER.TRUE_LR # Base LR for other parameters
        })

    # If no specific parameter groups are defined (e.g., all params treated equally or no trainable params),
    # use all trainable parameters for the optimizer. Otherwise, use the defined groups.
    optimizer_params = param_groups if param_groups else filter(lambda p: p.requires_grad, model.parameters())
        
    optimizer = build_optimizer(optimizer_params, config)
    # Scheduler can be None if not configured (e.g., config.SCHEDULER.TYPE is 'none')
    scheduler = build_scheduler(config, optimizer) 

    # --- Prepare components with Accelerator ---
    # This step wraps model, optimizer, dataloaders, and scheduler for distributed training,
    # device placement, and mixed precision.
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # --- Training Loop ---
    total_steps = 0 # Counter for total optimizer steps
    loguru_logger.info(f"Starting training for {args.num_epochs} epochs.")
    for epoch in range(args.num_epochs):
        model.train() # Set model to training mode
        epoch_start_time = time.time() # Record epoch start time for ETA calculation
        
        # Ensure train_dataloader is available (e.g., dataset path was valid)
        if train_dataloader is None:
            loguru_logger.error("Train dataloader is None. Skipping training epoch.")
            break # Or handle as appropriate (e.g., continue to next epoch)

        for step, batch in enumerate(train_dataloader):
            # Handle cases where collate_fn_skip_none might return None for an all-None batch
            if batch is None: 
                loguru_logger.warning(f"Skipping step {step} in epoch {epoch} due to empty batch after filtering.")
                continue
            
            # Learning rate warmup phase (if configured)
            # Warmup is applied per optimizer step.
            if config.TRAINER.WARMUP_STEP > 0 and total_steps < config.TRAINER.WARMUP_STEP:
                if config.TRAINER.WARMUP_TYPE == 'linear':
                    # Linear warmup from (WARMUP_RATIO * TRUE_LR) to TRUE_LR
                    base_lr = config.TRAINER.WARMUP_RATIO * config.TRAINER.TRUE_LR
                    current_lr_scale = total_steps / config.TRAINER.WARMUP_STEP
                    # For each param group, scale its target LR by current_lr_scale during warmup
                    for pg_idx, pg in enumerate(optimizer.param_groups):
                        # The 'lr' in param_groups is the target LR for that group (e.g., scaled for finetuning)
                        target_lr_for_group = param_groups[pg_idx]['lr'] if param_groups else config.TRAINER.TRUE_LR
                        pg['lr'] = base_lr + current_lr_scale * abs(target_lr_for_group - base_lr)

                # Add other warmup types (e.g., 'cosine') if necessary
            
            # Forward and backward pass with gradient accumulation
            # accelerator.accumulate handles the synchronization of gradients across processes
            # and enables/disables sync based on gradient_accumulation_steps.
            with accelerator.accumulate(model):
                # model_handler._trainval_inference performs the forward pass and computes loss,
                # updating the 'batch' dictionary in-place with 'loss' and 'loss_scalars'.
                model_handler._trainval_inference(batch) 
                loss = batch['loss'] # Total loss for backpropagation
                loss_scalars = batch['loss_scalars'] # Detailed breakdown of losses for logging

                accelerator.backward(loss) # Perform backpropagation

                # Gradient clipping and optimizer step occur only when gradients are synchronized
                # (i.e., after accumulating enough steps or on the last step of accumulation)
                if accelerator.sync_gradients:
                    if config.TRAINER.GRADIENT_CLIPPING > 0:
                        # Clip gradients to prevent exploding gradients
                        accelerator.clip_grad_norm_(model.parameters(), config.TRAINER.GRADIENT_CLIPPING)
                    
                    optimizer.step() # Update model weights
                    optimizer.zero_grad() # Reset gradients for the next iteration
                    total_steps += 1 # Increment total_steps only when optimizer.step() is called

            # Step-wise learning rate scheduler (if configured and gradients were synchronized)
            if scheduler is not None and scheduler['interval'] == 'step' and accelerator.sync_gradients:
                 # Unwrap scheduler if it was prepared by Accelerator and is a native PyTorch scheduler
                 unwrapped_scheduler = accelerator.unwrap_model(scheduler['scheduler'])
                 unwrapped_scheduler.step()

            # Log training metrics (on main process only)
            if total_steps > 0 and total_steps % args.log_every_n_steps == 0 and accelerator.is_main_process:
                log_dict = {"epoch": epoch, "step": total_steps, "train_loss": loss.item()}
                for k, v in loss_scalars.items(): # Log individual loss components
                    log_dict[f"train_{k}"] = v.item() if hasattr(v, 'item') else v
                if optimizer and hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                     log_dict["lr"] = optimizer.param_groups[0]['lr'] # Log current learning rate of the first param group
                accelerator.log(log_dict, step=total_steps) # Log to W&B

                # Calculate and print ETA for the current epoch
                # Note: `step` is the dataloader step, `total_steps` is the optimizer step.
                # ETA is based on dataloader steps within the current epoch.
                steps_in_epoch = len(train_dataloader)
                time_per_step_epoch = (time.time() - epoch_start_time) / (step + 1)
                remaining_steps_epoch = steps_in_epoch - (step + 1)
                eta_epoch_seconds = remaining_steps_epoch * time_per_step_epoch
                eta_epoch_str = time.strftime("%H:%M:%S", time.gmtime(eta_epoch_seconds))
                
                # Total optimizer steps across all epochs for progress tracking
                total_optimizer_steps_all_epochs = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_epochs
                
                accelerator.print(f"Epoch {epoch}, Step {total_steps}/{total_optimizer_steps_all_epochs}, Batch {step+1}/{steps_in_epoch}, ETA-Epoch: {eta_epoch_str}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6e}")

        # End of epoch scheduler step (if configured)
        if scheduler is not None and scheduler['interval'] == 'epoch':
            unwrapped_scheduler = accelerator.unwrap_model(scheduler['scheduler'])
            unwrapped_scheduler.step()

        # --- Validation Phase ---
        if epoch % args.val_every_n_epochs == 0:
            model.eval() # Set model to evaluation mode
            all_val_loss_scalars = [] # To store loss scalars from all validation batches across all processes
            all_val_metrics = []      # To store metrics from all validation batches across all processes
            
            loguru_logger.info(f"Running validation for epoch {epoch}...")
            
            if val_dataloader is None:
                loguru_logger.warning("Val dataloader is None. Skipping validation phase.")
            else:
                # Handle case where val_dataloader might be a list (for multiple validation datasets)
                current_val_loaders = val_dataloader if isinstance(val_dataloader, list) else [val_dataloader]

                for val_loader_idx, current_val_loader in enumerate(current_val_loaders):
                    if current_val_loader is None:
                        loguru_logger.warning(f"Validation dataloader at index {val_loader_idx} is None. Skipping.")
                        continue
                    
                    dataset_name_prefix = f"val_ds{val_loader_idx}_" if len(current_val_loaders) > 1 else "val_"

                    for val_step, val_batch in enumerate(current_val_loader):
                        if val_batch is None: # Handle collate_fn_skip_none
                            loguru_logger.warning(f"Skipping val_step {val_step} in val_loader {val_loader_idx} due to empty batch.")
                            continue
                        with torch.no_grad(): # Disable gradient calculations for validation
                            # Perform inference and get loss
                            model_handler._trainval_inference(val_batch) 
                            # Compute additional metrics
                            metrics_output = model_handler._compute_metrics(val_batch) 
                        
                        # The following explicit .to(accelerator.device) calls are defensive.
                        # If _trainval_inference and _compute_metrics always return tensors on accelerator.device,
                        # these might be redundant. accelerator.gather_for_metrics should handle device placement.
                        if 'loss_scalars' in val_batch and val_batch['loss_scalars'] is not None:
                            processed_loss_scalars = {}
                            if isinstance(val_batch['loss_scalars'], dict):
                                for key, tensor_val in val_batch['loss_scalars'].items():
                                    if isinstance(tensor_val, torch.Tensor):
                                        processed_loss_scalars[key] = tensor_val.to(accelerator.device)
                                    else: # Handle non-tensor values if any
                                        processed_loss_scalars[key] = torch.tensor(tensor_val, device=accelerator.device)
                            elif isinstance(val_batch['loss_scalars'], torch.Tensor): # Should be a dict, but handle tensor case
                                processed_loss_scalars = {'loss': val_batch['loss_scalars'].to(accelerator.device)}
                            val_batch['loss_scalars'] = processed_loss_scalars


                        if 'metrics' in metrics_output and metrics_output['metrics'] is not None:
                            processed_metrics = {}
                            if isinstance(metrics_output['metrics'], dict):
                                for key, tensor_val in metrics_output['metrics'].items():
                                    if isinstance(tensor_val, torch.Tensor):
                                        processed_metrics[key] = tensor_val.to(accelerator.device)
                                    else: # Handle non-tensor values
                                        processed_metrics[key] = torch.tensor(tensor_val, device=accelerator.device)
                            elif isinstance(metrics_output['metrics'], torch.Tensor): # Should be a dict
                                processed_metrics = {'metric': metrics_output['metrics'].to(accelerator.device)}
                            metrics_output['metrics'] = processed_metrics


                        # Gather loss scalars and metrics from all processes.
                        # `gather_for_metrics` is used as these are tensors intended for metric computation/logging.
                        # It handles gathering across all distributed processes.
                        gathered_loss_scalars = accelerator.gather_for_metrics(val_batch.get('loss_scalars', {}))
                        gathered_metrics = accelerator.gather_for_metrics(metrics_output.get('metrics', {}))
                        
                        # Prefix keys with dataset name if multiple validation sets
                        prefixed_gathered_loss_scalars = {f"{dataset_name_prefix}{k}": v for k, v in gathered_loss_scalars.items()}
                        prefixed_gathered_metrics = {f"{dataset_name_prefix}metric_{k}": v for k, v in gathered_metrics.items()}

                        all_val_loss_scalars.append(prefixed_gathered_loss_scalars)
                        all_val_metrics.append(prefixed_gathered_metrics)


                        # Visualize the first few validation batches on the main process
                        if accelerator.is_main_process and val_step < model_handler.n_vals_plot:
                            vis_prefix = f"{dataset_name_prefix}epoch{epoch}"
                            model_handler._visualize_batch(val_batch, val_batch, val_step, prefix=vis_prefix, global_step=total_steps)

            # Aggregate and log validation metrics (on main process only)
            if accelerator.is_main_process:
                # Combine all loss scalars and metrics into single lists of dictionaries
                flat_loss_scalars_list = [item for sublist in all_val_loss_scalars for item in ([sublist] if isinstance(sublist, dict) else sublist)]
                flat_metrics_list = [item for sublist in all_val_metrics for item in ([sublist] if isinstance(sublist, dict) else sublist)]

                # Aggregate by averaging
                avg_val_loss_scalars = {}
                if flat_loss_scalars_list:
                    # Get all unique keys from all dictionaries
                    all_keys = set(k for d in flat_loss_scalars_list for k in d.keys())
                    for key in all_keys:
                        # Concatenate tensors for the current key from all batches/processes, then mean
                        # Ensure that we only try to concatenate if the key exists and value is a tensor
                        valid_tensors = [d[key] for d in flat_loss_scalars_list if key in d and isinstance(d[key], torch.Tensor)]
                        if valid_tensors:
                             avg_val_loss_scalars[key] = torch.cat(valid_tensors).mean().item()


                avg_val_metrics = {}
                if flat_metrics_list:
                    all_keys = set(k for d in flat_metrics_list for k in d.keys())
                    for key in all_keys:
                        valid_tensors = [d[key] for d in flat_metrics_list if key in d and isinstance(d[key], torch.Tensor)]
                        if valid_tensors:
                            avg_val_metrics[key] = torch.cat(valid_tensors).mean().item()
                
                log_val_dict = {**avg_val_loss_scalars, **avg_val_metrics}
                if log_val_dict: # Only log if there's something to log
                    accelerator.log(log_val_dict, step=total_steps) # Log to W&B
                    accelerator.print(f"Validation Epoch {epoch} Results: {pprint.pformat(log_val_dict)}")

                    # Top-K checkpoint saving logic
                    current_epoch_metric_for_top_k = -float('inf')
                    selected_metric_name_for_top_k = "average_of_all_val_metrics"
                    
                    if avg_val_metrics:
                        # Calculate the average of all metric values in avg_val_metrics
                        metric_values = [v for v in avg_val_metrics.values() if isinstance(v, (int, float))]
                        if metric_values:
                            current_epoch_metric_for_top_k = sum(metric_values) / len(metric_values)
                            accelerator.print(f"Using metric '{selected_metric_name_for_top_k}' for top-K ranking: {current_epoch_metric_for_top_k:.4f}")
                        else:
                            accelerator.print("No numerical metric values found in avg_val_metrics to calculate an average for top-K ranking.")
                            current_epoch_metric_for_top_k = -float('inf') # Ensure it doesn't qualify if no metrics
                    else:
                        accelerator.print("avg_val_metrics is empty. Skipping top-K checkpoint logic.")

                    if current_epoch_metric_for_top_k > -float('inf'): # Proceed if a valid metric was calculated
                        save_dir = Path(args.output_dir) / args.exp_name
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Check if current model qualifies for top-K
                        if len(top_k_checkpoints) < K_BEST_CHECKPOINTS or current_epoch_metric_for_top_k > top_k_checkpoints[-1][0]:
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_ckpt_filename = f"best_avg_metric_{current_epoch_metric_for_top_k:.4f}_epoch_{epoch}_step_{total_steps}.ckpt"
                            best_ckpt_save_path = save_dir / best_ckpt_filename
                            
                            accelerator.save(unwrapped_model.state_dict(), str(best_ckpt_save_path))
                            accelerator.print(f"Saved new top-{K_BEST_CHECKPOINTS} checkpoint: {best_ckpt_save_path} with {selected_metric_name_for_top_k}: {current_epoch_metric_for_top_k:.4f}")
                            
                            top_k_checkpoints.append((current_epoch_metric_for_top_k, str(best_ckpt_save_path)))
                            top_k_checkpoints.sort(key=lambda x: x[0], reverse=True) # Sort by metric, highest first
                            
                            if len(top_k_checkpoints) > K_BEST_CHECKPOINTS:
                                removed_ckpt_info = top_k_checkpoints.pop() # Remove the worst one
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
            save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            ckpt_filename = f"epoch_{epoch}_step_{total_steps}.ckpt"
            save_path = save_dir / ckpt_filename
            
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), str(save_path))
            accelerator.print(f"Saved checkpoint (model weights) to {save_path}")

    # --- End of Training ---
    if accelerator.is_main_process:
        if profiler:
             profiler.summary() # Print profiler summary if enabled
        accelerator.end_training() # Clean up trackers (like W&B)
        loguru_logger.info("Training finished.")

if __name__ == "__main__":
    main()
