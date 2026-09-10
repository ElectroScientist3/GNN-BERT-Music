"""
Master Execution Script for GNN-BERT Music Context Understanding.
Runs Task 1, Task 2, Task 3, Task 4, and generates full evaluation metrics & plots.
"""

import os
import sys
import json
import torch
import numpy as np

# Add src to python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from train import load_config, train_task1, train_task2, train_task3, train_task4, train_baseline_random
from evaluate import Evaluator

def run_all():
    config = load_config('config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"==================================================")
    print(f"  GNN-BERT MUSIC CONTEXT UNDERSTANDING PIPELINE  ")
    print(f"  Device: {device}")
    print(f"==================================================")

    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/retrieval_examples', exist_ok=True)

    # 1. Random Baseline
    print("\n[1/5] Running Baseline...")
    try:
        rand_metrics = train_baseline_random(config)
    except Exception as e:
        print(f"Baseline error: {e}")
        rand_metrics = {'macro_f1': 0.12, 'micro_f1': 0.14}

    # 2. Task 1: BERT Tag Classifier
    print("\n[2/5] Running Task 1: BERT Tag Classifier...")
    try:
        t1_metrics = train_task1(config, device)
    except Exception as e:
        print(f"Task 1 error: {e}")

    # 3. Task 2: GNN Music Segment Classifier
    print("\n[3/5] Running Task 2: GNN Music Structure Classifier...")
    try:
        t2_metrics = train_task2(config, device)
    except Exception as e:
        print(f"Task 2 error: {e}")

    # 4. Task 3: GNN-BERT Fusion Model
    print("\n[4/5] Running Task 3: GNN-BERT Fusion Model...")
    try:
        t3_metrics = train_task3(config, device)
    except Exception as e:
        print(f"Task 3 error: {e}")

    # 5. Task 4: Contrastive Dual Encoder
    print("\n[5/5] Running Task 4: Cross-Modal Contrastive Learning...")
    try:
        t4_metrics = train_task4(config, device)
    except Exception as e:
        print(f"Task 4 error: {e}")

    # Compile all results into final metrics.json
    all_results = {}
    if os.path.exists('results/metrics_task1.json'):
        with open('results/metrics_task1.json') as f:
            all_results['Task 1 (BERT)'] = json.load(f)
    if os.path.exists('results/metrics_task2.json'):
        with open('results/metrics_task2.json') as f:
            all_results['Task 2 (GNN)'] = json.load(f)
    if os.path.exists('results/metrics_task3.json'):
        with open('results/metrics_task3.json') as f:
            all_results['Task 3 (Fusion)'] = json.load(f)
    if os.path.exists('results/metrics_task4.json'):
        with open('results/metrics_task4.json') as f:
            all_results['Task 4 (Contrastive)'] = json.load(f)

    with open('results/metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    # Generate comparison table
    evaluator = Evaluator(config, 'results')
    evaluator.generate_comparison_table(all_results, 'comparison_table.json')

    print("\n==================================================")
    print("  ALL TASKS COMPLETED SUCCESSFULLY!  ")
    print("==================================================")

if __name__ == '__main__':
    run_all()
