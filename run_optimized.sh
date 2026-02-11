#!/bin/bash
cd "/root/Carbon-aware Model training"
source venv/bin/activate
export PYTHONPATH="/root/Carbon-aware Model training/src:$PYTHONPATH"
python src/train.py configs/optimized.yaml