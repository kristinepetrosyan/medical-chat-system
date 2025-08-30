#!/bin/bash

# Medical Chat System - Training Script

echo " Medical Chat System - Training Script"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import torch, transformers, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required packages not installed. Please run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies check passed"

# Set default values
EPOCHS=${1:-3}
BATCH_SIZE=${2:-4}
LEARNING_RATE=${3:-5e-5}
DATA_FILE=${4:-""}

echo "🔧 Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Data File: $DATA_FILE"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed logs

# Run training
echo "🚀 Starting training..."
python training/train.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    ${DATA_FILE:+--data-file $DATA_FILE}

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved to: ./medical_chat_model"
    echo "Evaluation results saved to: evaluation_results.json"
else
    echo "Training failed!"
    exit 1
fi

echo ""
echo "Training script completed!"
echo "You can now run the chat system with:"
echo "  python scripts/run_chat.py --interactive"
echo "  python api/app.py"
