#!/bin/bash

# Script to visualize training progress from nohup.out

# Set default values
LOG_FILE="nohup.out"
OUTPUT_DIR="./training_plots"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-l LOG_FILE] [-o OUTPUT_DIR]"
            echo "  -l, --log LOG_FILE      Path to nohup.out file (default: nohup.out)"
            echo "  -o, --output OUTPUT_DIR Output directory for plots (default: ./training_plots)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if log file exists
if [[ ! -f "$LOG_FILE" ]]; then
    echo "Error: Log file '$LOG_FILE' not found!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Visualizing training progress from: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"

# Run the visualization script
python3 visualize_training_log.py --log_file "$LOG_FILE" --output_dir "$OUTPUT_DIR"

echo "Done! Check the plots in: $OUTPUT_DIR"
