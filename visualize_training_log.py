#!/usr/bin/env python3
"""
Script to visualize training metrics from nohup.out log file
"""

import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import argparse
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log and extract metrics"""
    
    training_data = []
    validation_data = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for training logs
    train_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{4}) INFO Epoch (\d+), Step (\d+)/\d+, Batch \d+/\d+, ETA: [\d:]+, Loss: ([\d.]+), LR_g0: ([\d.e-]+), LR_g1: ([\d.e-]+)'
    
    # Pattern for validation results
    val_pattern = r"Validation Epoch (\d+) Results: \{([^}]+)\}"
    
    # Extract training data
    for match in re.finditer(train_pattern, content):
        timestamp_str = match.group(1)
        epoch = int(match.group(2))
        step = int(match.group(3))
        loss = float(match.group(4))
        lr_g0 = float(match.group(5))
        lr_g1 = float(match.group(6))
        
        # Parse timestamp
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f%z')
        
        training_data.append({
            'timestamp': timestamp,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'lr_g0': lr_g0,
            'lr_g1': lr_g1
        })
    
    # Extract validation data
    for match in re.finditer(val_pattern, content):
        epoch = int(match.group(1))
        metrics_str = match.group(2)
        
        # Parse metrics from string
        metrics = {}
        metric_items = metrics_str.split(',')
        for item in metric_items:
            if ':' in item:
                key, value = item.split(':', 1)
                key = key.strip().strip("'\"")
                try:
                    value = float(value.strip())
                    metrics[key] = value
                except ValueError:
                    continue
        
        validation_data.append({
            'epoch': epoch,
            'metrics': metrics
        })
    
    return training_data, validation_data

def plot_training_curves(training_data, validation_data, output_dir='./plots'):
    """Plot training curves"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Convert to numpy arrays for easier plotting
    train_epochs = np.array([d['epoch'] for d in training_data])
    train_steps = np.array([d['step'] for d in training_data])
    train_losses = np.array([d['loss'] for d in training_data])
    train_lr_g0 = np.array([d['lr_g0'] for d in training_data])
    train_lr_g1 = np.array([d['lr_g1'] for d in training_data])
    train_timestamps = [d['timestamp'] for d in training_data]
    
    val_epochs = np.array([d['epoch'] for d in validation_data])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SAMatcher Training Progress', fontsize=16)
    
    # 1. Training Loss vs Steps
    axes[0, 0].plot(train_steps, train_losses, 'b-', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss vs Steps')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training Loss vs Epochs (averaged per epoch)
    unique_epochs = np.unique(train_epochs)
    epoch_losses = []
    for epoch in unique_epochs:
        epoch_mask = train_epochs == epoch
        epoch_losses.append(np.mean(train_losses[epoch_mask]))
    
    axes[0, 1].plot(unique_epochs, epoch_losses, 'r-o', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Training Loss')
    axes[0, 1].set_title('Training Loss vs Epochs')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rates
    axes[0, 2].plot(train_steps, train_lr_g0, 'g-', label='LR_g0', alpha=0.8)
    axes[0, 2].plot(train_steps, train_lr_g1, 'b-', label='LR_g1', alpha=0.8)
    axes[0, 2].set_xlabel('Training Steps')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rates vs Steps')
    axes[0, 2].set_yscale('log')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Validation Loss
    if validation_data:
        val_total_loss = []
        val_box_loss = []
        val_mask_loss = []
        
        for data in validation_data:
            metrics = data['metrics']
            val_total_loss.append(metrics.get('val_total_loss', None))
            val_box_loss.append(metrics.get('val_loss_box_total', None))
            val_mask_loss.append(metrics.get('val_loss_mask_total', None))
        
        # Filter out None values
        val_total_loss = [x for x in val_total_loss if x is not None]
        val_box_loss = [x for x in val_box_loss if x is not None]
        val_mask_loss = [x for x in val_mask_loss if x is not None]
        
        if val_total_loss:
            axes[1, 0].plot(val_epochs[:len(val_total_loss)], val_total_loss, 'ro-', label='Total Loss')
        if val_box_loss:
            axes[1, 0].plot(val_epochs[:len(val_box_loss)], val_box_loss, 'go-', label='Box Loss')
        if val_mask_loss:
            axes[1, 0].plot(val_epochs[:len(val_mask_loss)], val_mask_loss, 'bo-', label='Mask Loss')
            
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Validation Metrics (IoU scores)
    if validation_data:
        val_b_iou = []
        val_m_iou = []
        val_mb_iou = []
        val_b_oiou = []
        
        for data in validation_data:
            metrics = data['metrics']
            val_b_iou.append(metrics.get('val_metric_val_b_iou', None))
            val_m_iou.append(metrics.get('val_metric_val_m_iou', None))
            val_mb_iou.append(metrics.get('val_metric_val_mb_iou', None))
            val_b_oiou.append(metrics.get('val_metric_val_b_oiou', None))
        
        # Filter out None values and plot
        if any(x is not None for x in val_b_iou):
            filtered_b_iou = [(i, x) for i, x in enumerate(val_b_iou) if x is not None]
            if filtered_b_iou:
                epochs_b, vals_b = zip(*filtered_b_iou)
                axes[1, 1].plot([val_epochs[i] for i in epochs_b], vals_b, 'ro-', label='Box IoU')
        
        if any(x is not None for x in val_m_iou):
            filtered_m_iou = [(i, x) for i, x in enumerate(val_m_iou) if x is not None]
            if filtered_m_iou:
                epochs_m, vals_m = zip(*filtered_m_iou)
                axes[1, 1].plot([val_epochs[i] for i in epochs_m], vals_m, 'go-', label='Mask IoU')
        
        if any(x is not None for x in val_mb_iou):
            filtered_mb_iou = [(i, x) for i, x in enumerate(val_mb_iou) if x is not None]
            if filtered_mb_iou:
                epochs_mb, vals_mb = zip(*filtered_mb_iou)
                axes[1, 1].plot([val_epochs[i] for i in epochs_mb], vals_mb, 'bo-', label='Mask+Box IoU')
        
        if any(x is not None for x in val_b_oiou):
            filtered_b_oiou = [(i, x) for i, x in enumerate(val_b_oiou) if x is not None]
            if filtered_b_oiou:
                epochs_bo, vals_bo = zip(*filtered_b_oiou)
                axes[1, 1].plot([val_epochs[i] for i in epochs_bo], vals_bo, 'mo-', label='Box OIoU')
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU Score (%)')
        axes[1, 1].set_title('Validation IoU Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
    
    # 6. Training Progress over Time
    if len(train_timestamps) > 1:
        # Convert timestamps to hours from start
        start_time = train_timestamps[0]
        time_hours = [(ts - start_time).total_seconds() / 3600 for ts in train_timestamps]
        
        axes[1, 2].plot(time_hours, train_losses, 'b-', alpha=0.7, linewidth=0.8)
        axes[1, 2].set_xlabel('Time (hours)')
        axes[1, 2].set_ylabel('Training Loss')
        axes[1, 2].set_title('Training Loss vs Time')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    return fig

def print_training_summary(training_data, validation_data):
    """Print training summary statistics"""
    
    if not training_data:
        print("No training data found!")
        return
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Training stats
    total_steps = len(training_data)
    total_epochs = max([d['epoch'] for d in training_data]) + 1
    final_loss = training_data[-1]['loss']
    min_loss = min([d['loss'] for d in training_data])
    
    print(f"Total Steps: {total_steps}")
    print(f"Total Epochs: {total_epochs}")
    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Minimum Training Loss: {min_loss:.4f}")
    
    # Time stats
    if len(training_data) > 1:
        start_time = training_data[0]['timestamp']
        end_time = training_data[-1]['timestamp']
        duration = (end_time - start_time).total_seconds() / 3600
        print(f"Training Duration: {duration:.2f} hours")
        print(f"Steps per Hour: {total_steps / duration:.1f}")
    
    # Validation stats
    if validation_data:
        print(f"\nValidation Epochs: {len(validation_data)}")
        
        # Latest validation metrics
        latest_val = validation_data[-1]['metrics']
        print("\nLatest Validation Metrics:")
        for key, value in latest_val.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Visualize SAMatcher training log')
    parser.add_argument('--log_file', '-l', default='nohup.out', 
                       help='Path to nohup.out log file')
    parser.add_argument('--output_dir', '-o', default='outputs',
                       help='Output directory for plots')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not show plots (only save)')
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file '{log_file}' not found!")
        return
    
    print(f"Parsing log file: {log_file}")
    
    # Parse training log
    training_data, validation_data = parse_training_log(log_file)
    
    print(f"Found {len(training_data)} training steps")
    print(f"Found {len(validation_data)} validation epochs")
    
    if not training_data:
        print("No training data found in log file!")
        return
    
    # Print summary
    print_training_summary(training_data, validation_data)
    
    # Create plots
    if not args.no_show:
        plot_training_curves(training_data, validation_data, args.output_dir)
    else:
        # Modify matplotlib to not show plots
        import matplotlib
        matplotlib.use('Agg')
        plot_training_curves(training_data, validation_data, args.output_dir)

if __name__ == '__main__':
    main()
