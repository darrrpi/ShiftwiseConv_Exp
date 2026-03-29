import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime

from models import (
    ResNet18, ResNet18_SWConv, ConvNeXtTiny, 
    RepVGG_A0, MobileNetV3Small, EfficientNetB0
)
from utils import get_dataloaders, get_all_metrics, print_metrics_table


def create_model(model_name, num_classes, paths=8):
    """Create model by name"""
    models = {
        'resnet18': ResNet18(num_classes),
        'resnet18_swconv': ResNet18_SWConv(num_classes, paths=paths),
        'convnext_tiny': ConvNeXtTiny(num_classes),
        'repvgg_a0': RepVGG_A0(num_classes),
        'mobilenetv3_small': MobileNetV3Small(num_classes),
        'efficientnet_b0': EfficientNetB0(num_classes),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]


def train_model(model_name, dataset='cifar10', epochs=80, batch_size=128, paths=8):
    """Train a single model"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset}")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, num_classes = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        use_augmentation=True
    )
    
    # Create model
    model = create_model(model_name, num_classes, paths)
    
    # Measure metrics
    metrics = get_all_metrics(model, model_name, device)
    print_metrics_table([metrics])
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Simple training loop
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_{dataset}.pth")
            print(f"  -> New best model! Acc: {val_acc:.2f}%")
    
    print(f"\nFinished {model_name}! Best accuracy: {best_acc:.2f}%")
    
    return {
        'model': model_name,
        'dataset': dataset,
        'best_accuracy': best_acc,
        'params_m': metrics['params_m'],
        'flops_g': metrics['flops_g'],
        'inference_time_bs1_ms': metrics['inference_time_bs1_ms'],
        'inference_time_bs128_ms': metrics['inference_time_bs128_ms'],
    }


def main():
    # Models to train
    models = [ 'convnext_tiny', 
              'repvgg_a0', 'mobilenetv3_small', 'efficientnet_b0']
    
    all_results = []
    
    for model_name in models:
        result = train_model(model_name, dataset='cifar10', epochs=80, batch_size=128, paths=8)
        all_results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT A SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<25} {'Acc (%)':<10} {'Params (M)':<12} {'FLOPs (G)':<12} {'BS1 (ms)':<10} {'BS128 (ms)':<12}")
    print("-"*85)
    for r in all_results:
        print(f"{r['model']:<25} {r['best_accuracy']:<10.2f} {r['params_m']:<12.2f} "
              f"{r['flops_g']:<12.2f} {r['inference_time_bs1_ms']:<10.2f} {r['inference_time_bs128_ms']:<12.2f}")
    
    # Save results
    with open('experiment_A_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to experiment_A_summary.json")


if __name__ == '__main__':
    main()

