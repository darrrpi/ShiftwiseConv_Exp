import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_results(results_file='experiment_A_summary.json'):
    """Load experiment results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_accuracy_vs_params(results, save_path='results/plots/acc_vs_params.png'):
    """Plot Accuracy vs Number of Parameters"""
    models = [r['model'] for r in results]
    acc = [r['accuracy'] for r in results]
    params = [r['params (M)'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        plt.scatter(params[i], acc[i], s=200, c=[colors[i]], marker='o', label=model)
        plt.annotate(model, (params[i], acc[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Parameters (Millions)', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Model Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_accuracy_vs_flops(results, save_path='results/plots/acc_vs_flops.png'):
    """Plot Accuracy vs FLOPs"""
    models = [r['model'] for r in results]
    acc = [r['accuracy'] for r in results]
    flops = [r['FLOPs (G)'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        plt.scatter(flops[i], acc[i], s=200, c=[colors[i]], marker='s', label=model)
        plt.annotate(model, (flops[i], acc[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('FLOPs (Billions)', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Computational Cost', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_accuracy_vs_inference(results, save_path='results/plots/acc_vs_inference.png'):
    """Plot Accuracy vs Inference Time"""
    models = [r['model'] for r in results]
    acc = [r['accuracy'] for r in results]
    inference_time = [r['inference BS128 (ms)'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        plt.scatter(inference_time[i], acc[i], s=200, c=[colors[i]], marker='^', label=model)
        plt.annotate(model, (inference_time[i], acc[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Inference Time (ms, batch=128)', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Inference Speed', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_comparison_bar(results, save_path='results/plots/comparison_bar.png'):
    """Create a bar chart comparing all metrics"""
    models = [r['model'] for r in results]
    acc = [r['accuracy'] for r in results]
    params = [r['params (M)'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy bar chart
    colors = ['#2ecc71' if 'swconv' in m else '#3498db' for m in models]
    bars1 = ax1.bar(models, acc, color=colors)
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_ylim([min(acc)-5, max(acc)+5])
    
    # Add value labels on bars
    for bar, val in zip(bars1, acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Parameters bar chart
    bars2 = ax2.bar(models, params, color=colors)
    ax2.set_ylabel('Parameters (Millions)', fontsize=12)
    ax2.set_title('Model Size', fontsize=14)
    
    for bar, val in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}M', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def generate_all_plots():
    """Generate all comparison plots"""
    try:
        results = load_results('experiment_A_summary.json')
    except FileNotFoundError:
        print("Results file not found. Run train.py first to generate results.")
        return
    
    import os
    os.makedirs('results/plots', exist_ok=True)
    
    plot_accuracy_vs_params(results)
    plot_accuracy_vs_flops(results)
    plot_accuracy_vs_inference(results)
    plot_comparison_bar(results)
    
    print("Plots saved to results/plots/")

if __name__ == '__main__':
    generate_all_plots()