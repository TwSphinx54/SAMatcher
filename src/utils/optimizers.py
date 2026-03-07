import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, LambdaLR


def build_optimizer(params_or_model, config):
    """
    Builds an optimizer based on the provided configuration.

    Args:
        params_or_model (torch.nn.Module or iterable): Either the model instance
            or an iterable of parameters/parameter groups to optimize.
        config: Configuration object with optimizer settings.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    # Check if the input is a model or parameters/parameter groups
    if isinstance(params_or_model, torch.nn.Module):
        params = params_or_model.parameters()
    elif isinstance(params_or_model, (list, tuple)) or hasattr(params_or_model, '__iter__'):
        # Assumes it's an iterable of parameters or parameter groups
        params = params_or_model
    else:
        raise TypeError("Input to build_optimizer must be a model or an iterable of parameters/groups.")


    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Builds a learning rate scheduler based on the provided configuration.

    Args:
        config: Configuration object with scheduler settings.
        optimizer (torch.optim.Optimizer): The optimizer instance.

    Returns:
        dict or None: A dictionary containing the scheduler and its configuration,
                      or None if no scheduler is specified.
    """
    scheduler_name = config.TRAINER.SCHEDULER
    if not scheduler_name:
        return None

    lr_scheduler_configs = {
        'MultiStepLR': {
            'scheduler': MultiStepLR,
            'params': {
                'milestones': config.TRAINER.MSLR_MILESTONES,
                'gamma': config.TRAINER.MSLR_GAMMA,
            },
            'interval': 'epoch',
            'frequency': 1
        },
        'CosineAnnealingLR': {
            'scheduler': CosineAnnealingLR,
            'params': {
                'T_max': config.TRAINER.COSA_TMAX, # Changed from COSA_T_MAX
                'eta_min': config.TRAINER.COSA_ETA_MIN,
            },
            'interval': 'epoch', # Can be 'step' or 'epoch'
            'frequency': 1
        },
        'ExponentialLR': {
            'scheduler': ExponentialLR,
            'params': {
                'gamma': config.TRAINER.ELR_GAMMA,
            },
            'interval': 'epoch',
            'frequency': 1
        }
        # Add other schedulers as needed
    }

    if scheduler_name not in lr_scheduler_configs:
        raise ValueError(f"TRAINER.SCHEDULER = {scheduler_name} is not a valid scheduler!")

    cfg = lr_scheduler_configs[scheduler_name]
    
    # Add warmup if configured
    if config.TRAINER.WARMUP_STEP > 0 and config.TRAINER.WARMUP_TYPE == 'linear':
        # This creates a LambdaLR that applies linear warmup, then hands over to the main scheduler
        # Note: This is a simplified example. More complex combined schedulers might be needed.
        main_scheduler = cfg['scheduler'](optimizer, **cfg['params'])
        
        warmup_ratio = config.TRAINER.WARMUP_RATIO
        warmup_steps = config.TRAINER.WARMUP_STEP

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return warmup_ratio + (1.0 - warmup_ratio) * float(current_step) / float(max(1, warmup_steps))
            # After warmup, the behavior depends on how you want to combine it with the main scheduler.
            # This example assumes the main scheduler takes over completely.
            # For more complex scenarios (e.g., main_scheduler.step() called based on epoch),
            # this lambda might need to be more sophisticated or a ChainedScheduler used.
            # For simplicity, this lambda is for step-based warmup.
            # If main_scheduler is epoch-based, this interaction needs careful design.
            return 1.0 # Placeholder, as main_scheduler will be stepped separately.

        # This LambdaLR is primarily for the warmup phase.
        # The main scheduler (MultiStepLR, CosineAnnealingLR, etc.) will be stepped according to its interval.
        # Accelerate's prepare will handle the optimizer and scheduler.
        # The interaction between a warmup LambdaLR (stepped per step) and a main scheduler (stepped per epoch)
        # needs to be managed in the training loop. Accelerate doesn't automatically chain them in a specific way.
        # The current main.py logic handles linear warmup by directly modifying optimizer.param_groups.
        # So, we might just return the main scheduler here if warmup is handled externally.
        
        # Given main.py's current warmup logic, we just return the main scheduler.
        # If warmup was to be integrated *into* the scheduler object returned here,
        # it would require a more complex scheduler like ChainedScheduler or custom lambda.
        scheduler_instance = main_scheduler
        
    else:
        scheduler_instance = cfg['scheduler'](optimizer, **cfg['params'])

    return {
        'scheduler': scheduler_instance,
        'interval': cfg['interval'], # 'step' or 'epoch'
        'frequency': cfg['frequency'], # step every 'frequency' steps/epochs
        'name': scheduler_name, # Optional: for logging
    }
