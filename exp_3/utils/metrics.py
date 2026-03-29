import torch
import time
import numpy as np
from thop import profile

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, input_size=(1, 3, 32, 32), device='cuda'):
    """Measure FLOPs using thop"""
    model.eval()
    
    # Move model to device
    model = model.to(device)
    
    input_tensor = torch.randn(input_size).to(device)
    
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params


def measure_inference_time(model, batch_size=128, num_runs=100, device='cuda'):
    """Measure inference time for given batch size"""
    model.eval()
    
    # Move model to device
    model = model.to(device)
    
    # Warmup
    input_tensor = torch.randn(batch_size, 3, 32, 32).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    return avg_time, std_time


def get_all_metrics(model, model_name, device='cuda', input_size=(1, 3, 32, 32)):
    """Get all metrics for a model"""
    model.eval()
    
    # Move model to device
    model = model.to(device)
    
    params = count_parameters(model)
    flops, _ = measure_flops(model, input_size, device)
    
    # Inference time for batch size 1
    time_bs1, _ = measure_inference_time(model, batch_size=1, device=device)
    
    # Inference time for batch size 128
    time_bs128, std_bs128 = measure_inference_time(model, batch_size=128, device=device)
    
    return {
        'model': model_name,
        'params': params,
        'params_m': params / 1e6,
        'flops': flops,
        'flops_g': flops / 1e9,
        'inference_time_bs1_ms': time_bs1,
        'inference_time_bs128_ms': time_bs128,
        'inference_time_bs128_std_ms': std_bs128,
    }


def print_metrics_table(metrics_list):
    """Print metrics in a formatted table"""
    print("\n" + "="*100)
    print(f"{'Model':<25} {'Params (M)':<12} {'FLOPs (G)':<12} {'Time BS1 (ms)':<15} {'Time BS128 (ms)':<15}")
    print("="*100)
    
    for m in metrics_list:
        print(f"{m['model']:<25} {m['params_m']:<12.2f} {m['flops_g']:<12.2f} "
              f"{m['inference_time_bs1_ms']:<15.2f} {m['inference_time_bs128_ms']:<15.2f}")
    print("="*100)