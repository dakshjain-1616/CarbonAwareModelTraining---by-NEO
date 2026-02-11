# Carbon-Aware Model Training Pipeline

A comprehensive PyTorch-based training pipeline that optimizes compute scheduling based on electricity carbon intensity, reduces GPU utilization through gradient accumulation, and tracks carbon emissions throughout the training process.

## Architecture

```
```
├── src/
│   ├── scheduler.py      # Carbon intensity API integration & scheduling logic
│   ├── tracker.py        # CodeCarbon emissions tracking wrapper
│   ├── train.py          # Main training pipeline with gradient accumulation
│   └── utils.py          # Configuration loading and logging utilities
├── configs/
│   ├── baseline.yaml     # Configuration for baseline training
│   └── optimized.yaml    # Configuration for carbon-aware optimized training
├── output/               # Training outputs, logs, and emissions data
├── models/               # Saved model checkpoints
├── data/                 # MNIST dataset (auto-downloaded)
└── requirements.txt      # Python dependencies
```
```

## Features

1. **Carbon-Aware Scheduling**
   - Real-time carbon intensity monitoring (with mock fallback)
   - Configurable intensity thresholds
   - Training delay until low-carbon windows
   - Robust error handling with fallback mechanisms

2. **Gradient Accumulation**
   - Reduces GPU memory footprint
   - Maintains effective batch size
   - Configurable accumulation steps (1, 2, 4, 8, 16...)
   - Preserves training convergence

3. **Emissions Tracking**
   - Real-time CO2 emissions monitoring via CodeCarbon
   - Energy consumption tracking (kWh)
   - Power consumption metrics (Watts)
   - Comprehensive JSON reports

4. **Modular Design**
   - YAML-based configuration
   - Separate scheduler, tracker, and trainer components
   - Easy to extend and customize
   - Support for different model architectures

## How NEO Tackled This Task

This carbon-aware training pipeline was architected with modularity and extensibility as core principles:

### Architectural Decisions

**1. Scheduler/Tracker Separation**
- **Scheduler Module** (`src/scheduler.py`): Isolated carbon intensity API integration with graceful fallback to mock data when APIs are unavailable. This design ensures the pipeline remains operational even without live API access, using simulated diurnal patterns that mirror real-world carbon intensity fluctuations.
- **Tracker Module** (`src/tracker.py`): Dedicated wrapper around CodeCarbon for emissions monitoring, providing clean interfaces for starting/stopping tracking and extracting metrics. This separation allows easy swapping of tracking backends.

**2. Gradient Accumulation Strategy**
- Implemented in `src/train.py` with configurable accumulation steps (2, 4, 8, 16)
- Reduces GPU memory footprint by processing smaller micro-batches while maintaining effective batch size
- Critical for training large models on memory-constrained hardware
- Convergence validation ensures model quality is preserved

**3. Configuration-Driven Design**
- YAML-based configuration files (`configs/baseline.yaml`, `configs/optimized.yaml`) 
- Allows A/B testing of carbon-aware vs. standard training without code changes
- Clear separation between baseline (no scheduling delays) and optimized (carbon-aware) runs

**4. Robust Fallback Mechanisms**
- API failures trigger automatic fallback to synthetic carbon intensity data
- Ensures training can proceed even in offline/restricted network environments
- Logs all fallback events for transparency

**5. Comprehensive Logging & Reporting**
- Timestamped training logs capture all scheduling decisions
- JSON comparison reports quantify carbon savings
- CSV emissions logs enable post-hoc analysis

### Key Implementation Highlights
- Mock carbon intensity data uses realistic sinusoidal patterns (peak at 18:00, trough at 03:00)
- Scheduler evaluates windows every 10 minutes, comparing intensity against configurable thresholds
- Training checkpoints saved independently for baseline and optimized runs
- Modular design allows easy extension with custom schedulers or trackers

## Future Roadmap: Possible Enhancements

To make this pipeline production-ready and more powerful, consider these feature additions:

### 1. **Adaptive Batch Sizing**
- Dynamically adjust batch size based on real-time GPU memory availability
- Automatically scale gradient accumulation steps to maintain throughput
- Monitor GPU utilization and adapt to maximize efficiency

### 2. **Real-Time Monitoring Dashboard**
- Web-based dashboard (Flask/Streamlit) showing live training metrics
- Real-time carbon intensity visualization with forecasts
- Interactive controls to pause/resume training based on carbon thresholds
- Integration with Weights & Biases or MLflow for experiment tracking

### 3. **Carbon Budget Early Stopping**
- Set maximum carbon budget (kgCO2) for training runs
- Automatically halt training when budget exhausted
- Trade-off visualization: accuracy vs. carbon footprint

### 4. **Multi-Region Carbon Arbitrage**
- Support distributed training across multiple data centers
- Route compute to regions with lowest current carbon intensity
- Coordinate workload migration based on geographic carbon forecasts

### 5. **Predictive Scheduling with Forecasts**
- Integrate carbon intensity forecast APIs (24-48 hour predictions)
- Plan training schedules in advance to maximize low-carbon windows
- Automatically pause long-running jobs when high-carbon periods predicted

### 6. **Hyperparameter Optimization for Carbon Efficiency**
- Bayesian optimization of (accuracy, carbon) multi-objective
- AutoML integration to find Pareto-optimal configurations
- Quantify accuracy degradation acceptable for carbon savings

### 7. **Integration with Cloud Carbon Metrics**
- Native support for AWS Carbon Footprint Tool, Google Cloud Carbon Footprint
- Azure Sustainability Calculator integration
- Unified carbon reporting across on-prem and cloud

### 8. **Enhanced Visualization Suite** ✓ *Implemented*
- Comparative emissions heatmaps across experiments
- Carbon intensity vs. training throughput correlation analysis
- Marginal carbon cost per accuracy point gained
- Geographic carbon intensity maps for distributed training decisions

### 9. **Model Compression Pipelines**
- Post-training quantization to reduce inference carbon cost
- Knowledge distillation to smaller, more efficient models
- Pruning strategies evaluated against carbon metrics

### 10. **Policy-Based Scheduling**
- Define organizational carbon policies (e.g., "never train when intensity >400 gCO2/kWh")
- Compliance reporting for corporate sustainability goals
- Integration with carbon offset procurement systems

### 11. **Checkpoint-Based Preemption**
- Gracefully pause training during high-carbon periods
- Resume from checkpoints when intensity drops
- Minimize wasted compute from hard interruptions

### 12. **Carbon-Aware Data Loading**
- Prefetch datasets during low-carbon periods
- Schedule data preprocessing tasks separately from training
- Reduce I/O bottlenecks during critical training windows

These enhancements would transform this from a demonstration pipeline into an enterprise-grade carbon-aware MLOps platform.

## Setup

```bash
cd "/root/Carbon-aware Model training"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run Baseline Training (no optimization)
```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/train.py configs/baseline.yaml
```

### Run Optimized Training (carbon-aware + gradient accumulation)
```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/train.py configs/optimized.yaml
```

### Generate Comparison Report
```bash
source venv/bin/activate
python generate_comparison.py
```

## Configuration

### Scheduler Configuration
```yaml
scheduler:
  enabled: true                    # Enable/disable carbon-aware scheduling
  use_mock_data: true             # Use mock data (true) or real API (false)
  carbon_threshold: 300           # Carbon intensity threshold (gCO2/kWh)
  max_wait_seconds: 3600          # Maximum wait time for low-carbon window
  check_interval: 300             # Seconds between carbon intensity checks
  wait_for_low_carbon: true       # Wait for low-carbon window before starting
```

### Training Configuration
```yaml
training:
  run_name: optimized
  batch_size: 16                  # Per-step batch size
  gradient_accumulation_steps: 4  # Accumulation steps (effective_batch = 64)
  epochs: 3
  learning_rate: 0.001
  model_save_path: model.pt
```

## Output Files

- `output/summary_<run_name>.json` - Complete training summary with metrics
- `output/comparison_report.json` - Side-by-side comparison of runs
- `output/emissions.csv` - Detailed CodeCarbon emissions log
- `models/model_<run_name>.pt` - Trained model checkpoint
- `output/training_<timestamp>.log` - Detailed training logs

## Comparison Metrics

The comparison report includes:
- CO2 emissions (kg) and reduction percentage
- GPU memory usage and reduction
- Training accuracy comparison
- Scheduler delay times
- Convergence validation

## GPU Support

The pipeline automatically detects and uses CUDA GPUs when available:
- Mixed precision training (FP16)
- Pin memory optimization
- Memory-efficient DataLoader settings
- Falls back gracefully to CPU if needed

## Extending the Pipeline

### Adding New Models
Modify `src/train.py` and replace `SimpleCNN` with your architecture.

### Custom Carbon APIs
Update `src/scheduler.py` `get_carbon_intensity()` with your API endpoint.

### Different Datasets
Modify `prepare_data()` in `src/train.py` to load your dataset.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CodeCarbon 2.3+
- CUDA (optional, for GPU acceleration)

## License

MIT