import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def load_emissions_data(emissions_csv_path):
    df = pd.read_csv(emissions_csv_path)
    return df

def load_training_logs(log_dir):
    log_files = list(Path(log_dir).glob("training_*.log"))
    training_data = {"baseline": [], "optimized": []}
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            mode = "baseline" if "baseline" in log_file.name else "optimized"
            for line in lines:
                if "Epoch" in line and "Loss" in line:
                    try:
                        parts = line.split("Loss:")
                        if len(parts) > 1:
                            loss = float(parts[1].strip().split()[0])
                            training_data[mode].append(loss)
                    except:
                        pass
    
    return training_data

def plot_carbon_intensity_vs_power(emissions_df, output_dir):
    plt.figure(figsize=(12, 6))
    
    if 'carbon_intensity' in emissions_df.columns and 'power' in emissions_df.columns:
        plt.subplot(1, 2, 1)
        plt.plot(emissions_df.index, emissions_df['carbon_intensity'], 
                 label='Carbon Intensity', color='green', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Carbon Intensity (gCO2/kWh)')
        plt.title('Carbon Intensity Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(emissions_df.index, emissions_df['power'], 
                 label='Power Usage', color='red', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Power (Watts)')
        plt.title('Power Consumption Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Carbon Intensity or Power data not available', 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'carbon_intensity_vs_power.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_cumulative_emissions(emissions_df, output_dir):
    plt.figure(figsize=(10, 6))
    
    if 'emissions' in emissions_df.columns:
        cumulative_emissions = emissions_df['emissions'].cumsum()
        plt.plot(emissions_df.index, cumulative_emissions, 
                 label='Cumulative CO2 Emissions', color='darkred', linewidth=2.5)
        plt.fill_between(emissions_df.index, cumulative_emissions, alpha=0.3, color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative CO2 Emissions (kg)')
        plt.title('Cumulative Carbon Emissions During Training')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        final_emissions = cumulative_emissions.iloc[-1]
        plt.text(0.7, 0.95, f'Total: {final_emissions:.6f} kg CO2', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'Emissions data not available', 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cumulative_emissions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_training_loss_comparison(training_data, output_dir):
    plt.figure(figsize=(10, 6))
    
    if training_data['baseline'] and training_data['optimized']:
        epochs_baseline = range(1, len(training_data['baseline']) + 1)
        epochs_optimized = range(1, len(training_data['optimized']) + 1)
        
        plt.plot(epochs_baseline, training_data['baseline'], 
                label='Baseline Training', marker='o', linewidth=2, color='blue')
        plt.plot(epochs_optimized, training_data['optimized'], 
                label='Carbon-Aware Training', marker='s', linewidth=2, color='green')
        
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Comparison: Baseline vs Carbon-Aware')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Training loss data not available', 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_loss_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_comparison_summary(comparison_report_path, output_dir):
    with open(comparison_report_path, 'r') as f:
        report = json.load(f)
    
    baseline = report.get('baseline', {})
    optimized = report.get('optimized', {})
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    categories = ['CO2 Emissions\n(kg)', 'Energy Consumed\n(kWh)', 'Training Time\n(seconds)']
    baseline_values = [
        baseline.get('total_emissions_kg', 0),
        baseline.get('energy_consumed_kwh', 0),
        baseline.get('training_time_seconds', 0)
    ]
    optimized_values = [
        optimized.get('total_emissions_kg', 0),
        optimized.get('energy_consumed_kwh', 0),
        optimized.get('training_time_seconds', 0)
    ]
    
    x = range(len(categories))
    width = 0.35
    
    for idx, (cat, base_val, opt_val) in enumerate(zip(categories, baseline_values, optimized_values)):
        axes[idx].bar(['Baseline', 'Carbon-Aware'], [base_val, opt_val], 
                     color=['#FF6B6B', '#4ECDC4'], width=0.6)
        axes[idx].set_title(cat, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
        
        if base_val > 0:
            reduction = ((base_val - opt_val) / base_val) * 100
            axes[idx].text(0.5, max(base_val, opt_val) * 1.05, 
                          f'{reduction:.1f}% reduction' if reduction > 0 else f'{abs(reduction):.1f}% increase',
                          ha='center', fontsize=10, color='darkgreen' if reduction > 0 else 'darkred')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    emissions_csv = output_dir / "emissions.csv"
    comparison_report = output_dir / "comparison_report.json"
    
    print("=== Carbon-Aware Training Visualization Generator ===\n")
    
    if emissions_csv.exists():
        print(f"Loading emissions data from: {emissions_csv}")
        emissions_df = load_emissions_data(emissions_csv)
        print(f"Loaded {len(emissions_df)} emission records\n")
        
        print("Generating visualizations...")
        plot_carbon_intensity_vs_power(emissions_df, output_dir)
        plot_cumulative_emissions(emissions_df, output_dir)
    else:
        print(f"Warning: {emissions_csv} not found. Skipping emissions plots.\n")
    
    print("\nLoading training logs...")
    training_data = load_training_logs(output_dir)
    plot_training_loss_comparison(training_data, output_dir)
    
    if comparison_report.exists():
        print(f"\nLoading comparison report from: {comparison_report}")
        plot_comparison_summary(comparison_report, output_dir)
    else:
        print(f"Warning: {comparison_report} not found. Skipping comparison summary.\n")
    
    print("\n=== Visualization Complete ===")
    print(f"All charts saved to: {output_dir}")

if __name__ == "__main__":
    main()