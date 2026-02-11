import json
from pathlib import Path
from datetime import datetime

output_dir = Path('./output')

baseline_summary = output_dir / 'summary_baseline.json'
optimized_summary = output_dir / 'summary_optimized.json'

if not baseline_summary.exists() or not optimized_summary.exists():
    print("ERROR: Missing summary files")
    exit(1)

with open(baseline_summary) as f:
    baseline = json.load(f)

with open(optimized_summary) as f:
    optimized = json.load(f)

comparison = {
    'generated_at': datetime.now().isoformat(),
    'baseline': {
        'run_name': baseline['config']['run_name'],
        'emissions_kg_co2': baseline['emissions'].get('emissions_kg_co2', 0),
        'batch_size': baseline['config']['batch_size'],
        'accumulation_steps': baseline['config']['gradient_accumulation_steps'],
        'effective_batch_size': baseline['config']['batch_size'] * baseline['config']['gradient_accumulation_steps'],
        'final_loss': baseline['training_metrics']['final_loss'],
        'test_loss': baseline['training_metrics']['test_loss'],
        'test_accuracy': baseline['training_metrics']['test_accuracy'],
        'max_memory_gb': baseline['training_metrics']['max_memory_gb'],
        'avg_epoch_time': baseline['training_metrics']['avg_epoch_time'],
        'scheduler_wait_time': baseline['schedule_decision']['wait_time_seconds'],
        'carbon_intensity': baseline['schedule_decision'].get('final_intensity'),
        'device': baseline['training_metrics']['device']
    },
    'optimized': {
        'run_name': optimized['config']['run_name'],
        'emissions_kg_co2': optimized['emissions'].get('emissions_kg_co2', 0),
        'batch_size': optimized['config']['batch_size'],
        'accumulation_steps': optimized['config']['gradient_accumulation_steps'],
        'effective_batch_size': optimized['config']['batch_size'] * optimized['config']['gradient_accumulation_steps'],
        'final_loss': optimized['training_metrics']['final_loss'],
        'test_loss': optimized['training_metrics']['test_loss'],
        'test_accuracy': optimized['training_metrics']['test_accuracy'],
        'max_memory_gb': optimized['training_metrics']['max_memory_gb'],
        'avg_epoch_time': optimized['training_metrics']['avg_epoch_time'],
        'scheduler_wait_time': optimized['schedule_decision']['wait_time_seconds'],
        'carbon_intensity': optimized['schedule_decision'].get('final_intensity'),
        'device': optimized['training_metrics']['device']
    },
    'comparison': {
        'emissions_reduction_percent': ((baseline['emissions'].get('emissions_kg_co2', 0) - optimized['emissions'].get('emissions_kg_co2', 0)) / baseline['emissions'].get('emissions_kg_co2', 0.0001)) * 100,
        'memory_reduction_gb': baseline['training_metrics']['max_memory_gb'] - optimized['training_metrics']['max_memory_gb'],
        'memory_reduction_percent': ((baseline['training_metrics']['max_memory_gb'] - optimized['training_metrics']['max_memory_gb']) / baseline['training_metrics']['max_memory_gb']) * 100 if baseline['training_metrics']['max_memory_gb'] > 0 else 0,
        'accuracy_difference': optimized['training_metrics']['test_accuracy'] - baseline['training_metrics']['test_accuracy'],
        'scheduler_delay_seconds': optimized['schedule_decision']['wait_time_seconds'],
        'effective_batch_size_same': baseline['config']['batch_size'] * baseline['config']['gradient_accumulation_steps'] == optimized['config']['batch_size'] * optimized['config']['gradient_accumulation_steps']
    },
    'summary': {
        'baseline_completed': True,
        'optimized_completed': True,
        'gradient_accumulation_working': optimized['config']['gradient_accumulation_steps'] > 1,
        'carbon_scheduler_working': optimized['schedule_decision']['wait_time_seconds'] > 0 or optimized['schedule_decision'].get('checks_performed', 0) > 0,
        'both_converged': baseline['training_metrics']['final_loss'] < 1.0 and optimized['training_metrics']['final_loss'] < 1.0
    }
}

output_path = output_dir / 'comparison_report.json'
with open(output_path, 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\n{'='*60}")
print("COMPARISON REPORT GENERATED")
print(f"{'='*60}\n")
print(f"Baseline Emissions: {comparison['baseline']['emissions_kg_co2']:.6f} kg CO2")
print(f"Optimized Emissions: {comparison['optimized']['emissions_kg_co2']:.6f} kg CO2")
print(f"Emissions Reduction: {comparison['comparison']['emissions_reduction_percent']:.2f}%")
print(f"\nBaseline Memory: {comparison['baseline']['max_memory_gb']:.2f} GB")
print(f"Optimized Memory: {comparison['optimized']['max_memory_gb']:.2f} GB")
print(f"Memory Reduction: {comparison['comparison']['memory_reduction_gb']:.2f} GB ({comparison['comparison']['memory_reduction_percent']:.2f}%)")
print(f"\nBaseline Accuracy: {comparison['baseline']['test_accuracy']:.2f}%")
print(f"Optimized Accuracy: {comparison['optimized']['test_accuracy']:.2f}%")
print(f"Accuracy Difference: {comparison['comparison']['accuracy_difference']:.2f}%")
print(f"\nScheduler Wait Time: {comparison['optimized']['scheduler_wait_time']:.1f}s")
print(f"\nReport saved to: {output_path}")
print(f"{'='*60}\n")