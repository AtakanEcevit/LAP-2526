"""
Full benchmark runner.
Runs all experiment configurations and produces comparison tables.

Usage: python -m evaluation.benchmark
"""

import os
import sys
import csv
import yaml
import json
import glob
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_configs(configs_dir="configs"):
    """Find all YAML config files."""
    return sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))


def generate_comparison_table(results, output_dir="results"):
    """
    Generate CSV and console table from benchmark results.
    
    Args:
        results: list of dicts with keys:
            model, modality, dataset, k_shot, accuracy, eer, far, frr, auc, d_prime
    """
    os.makedirs(output_dir, exist_ok=True)

    # CSV output
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    fieldnames = ['model', 'modality', 'dataset', 'k_shot',
                  'accuracy', 'eer', 'far', 'frr', 'auc', 'd_prime']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                             for k, v in row.items() if k in fieldnames})

    # Console table
    print(f"\n{'='*100}")
    print(f"  BENCHMARK RESULTS — Siamese vs Prototypical Networks")
    print(f"{'='*100}")
    print(f"{'Model':>15s} | {'Modality':>12s} | {'Dataset':>10s} | "
          f"{'K-Shot':>6s} | {'Acc':>7s} | {'EER':>7s} | "
          f"{'FAR':>7s} | {'FRR':>7s} | {'AUC':>7s} | {'d\'':>7s}")
    print("-" * 100)

    for row in results:
        print(f"{row['model']:>15s} | {row['modality']:>12s} | "
              f"{row['dataset']:>10s} | {row['k_shot']:>6s} | "
              f"{row['accuracy']:>7.4f} | {row['eer']:>7.4f} | "
              f"{row['far']:>7.4f} | {row['frr']:>7.4f} | "
              f"{row['auc']:>7.4f} | {row['d_prime']:>7.4f}")

    print(f"{'='*100}")
    print(f"\n  CSV saved to: {csv_path}")

    # JSON output for programmatic access
    json_path = os.path.join(output_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"  JSON saved to: {json_path}")

    # LaTeX table
    latex_path = os.path.join(output_dir, 'comparison_table.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Biometric Verification: Siamese vs Prototypical Networks}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lllcccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Modality & K-Shot & Acc & EER & FAR & FRR & AUC & d' \\\\\n")
        f.write("\\midrule\n")
        for row in results:
            f.write(f"{row['model']} & {row['modality']} & {row['k_shot']} & "
                    f"{row['accuracy']:.4f} & {row['eer']:.4f} & "
                    f"{row['far']:.4f} & {row['frr']:.4f} & "
                    f"{row['auc']:.4f} & {row['d_prime']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  LaTeX saved to: {latex_path}")


def main():
    """
    Run full benchmark.
    
    This script expects models to already be trained.
    It loads each checkpoint and evaluates across k-shot values.
    """
    print("=" * 60)
    print("  Full Benchmark Runner")
    print("  Run `python train.py --config <config>` first for each config")
    print("=" * 60)

    configs = find_configs()
    if not configs:
        print("\n  No config files found in configs/ directory.")
        print("  Run training first with: python train.py --config <config.yaml>")
        return

    print(f"\n  Found {len(configs)} configurations:")
    for c in configs:
        print(f"    - {os.path.basename(c)}")

    print(f"\n  To run the full benchmark, train all models first,")
    print(f"  then evaluate each with evaluate.py.")
    print(f"  This script will be extended to auto-run all evaluations.")


if __name__ == "__main__":
    main()
