"""
L1 Pruning Demo Script

This script demonstrates how to apply L1 pruning to a SegFormer model.
It shows the complete workflow including:
1. Loading a model
2. Applying L1 pruning
3. Validating the pruned model
4. Comparing statistics before and after pruning
"""

import argparse
import datetime
import torch
import torch.nn as nn
from nets.segformer import SegFormer
from nets.pruning import L1Pruner, validate_pruned_model, count_parameters


def print_separator(title=""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}\n")


def print_model_info(model, title="Model Information"):
    """Print model information including parameters and structure."""
    print_separator(title)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Count Conv2d layers
    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"Conv2d layers: {conv_count}")
    
    # Count total channels in Conv2d layers
    total_channels = sum(m.weight.shape[0] for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"Total Conv2d output channels: {total_channels}")


def compare_models(original_model, pruned_model):
    """Compare original and pruned models."""
    print_separator("Model Comparison")
    
    orig_params, _ = count_parameters(original_model)
    pruned_params, _ = count_parameters(pruned_model)
    
    print(f"Original parameters: {orig_params:,}")
    print(f"Pruned parameters: {pruned_params:,}")
    print(f"Reduction: {orig_params - pruned_params:,} ({(1 - pruned_params/orig_params)*100:.2f}%)")
    
    # Note: In soft pruning, parameter count may not change significantly
    # because we zero out weights rather than removing them
    print("\nNote: This demo uses 'soft pruning' (zeroing weights).")
    print("For true parameter reduction, implement model-specific rebuilding.")


def demo_l1_pruning(
    num_classes=21,
    phi='b0',
    pruning_ratio=0.3,
    input_shape=(1, 3, 512, 512),
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Demonstrate L1 pruning on a SegFormer model.
    
    Args:
        num_classes: Number of output classes
        phi: Model variant (b0-b5)
        pruning_ratio: Ratio of channels to prune (0.0-1.0)
        input_shape: Input tensor shape for validation
        device: Device to run on ('cuda' or 'cpu')
    """
    
    print_separator("L1 Pruning Demo for SegFormer")
    print(f"Configuration:")
    print(f"  Model: SegFormer-{phi}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Pruning ratio: {pruning_ratio}")
    print(f"  Device: {device}")
    print(f"  Input shape: {input_shape}")
    
    # Step 1: Create and load model
    print_separator("Step 1: Creating Model")
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False)
    model = model.to(device)
    model.eval()
    
    print_model_info(model, "Original Model")
    
    # Step 2: Validate original model
    print_separator("Step 2: Validating Original Model")
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        original_output = model(dummy_input)
    print(f"✓ Original model forward pass successful")
    print(f"  Output shape: {original_output.shape}")
    
    # Step 3: Apply L1 Pruning
    print_separator("Step 3: Applying L1 Pruning")
    pruner = L1Pruner(model, pruning_ratio=pruning_ratio)
    pruned_model = pruner.prune()
    
    # Step 4: Get pruning statistics
    print_separator("Step 4: Pruning Statistics")
    stats = pruner.get_pruning_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Step 5: Validate pruned model
    print_separator("Step 5: Validating Pruned Model")
    validation_success = validate_pruned_model(pruned_model, input_shape)
    
    if validation_success:
        with torch.no_grad():
            pruned_output = pruned_model(dummy_input)
        print(f"  Output shape: {pruned_output.shape}")
        
        # Check output difference
        if isinstance(original_output, tuple):
            original_logits = original_output[0]
        else:
            original_logits = original_output
            
        if isinstance(pruned_output, tuple):
            pruned_logits = pruned_output[0]
        else:
            pruned_logits = pruned_output
            
        output_diff = torch.mean(torch.abs(original_logits - pruned_logits)).item()
        print(f"  Average output difference: {output_diff:.6f}")
        
    # Step 6: Compare models
    compare_models(model, pruned_model)
    
    # Step 7: Summary
    print_separator("Summary")
    print("✓ L1 pruning demo completed successfully!")
    print(f"✓ Pruned {stats['pruned_channels']} out of {stats['total_channels']} channels")
    print(f"✓ Pruned model can perform forward pass")
    print(f"\nThe pruned model is ready for fine-tuning or deployment.")
    print(f"For best results, fine-tune the pruned model on your dataset.")
    print_separator()
    
    return pruned_model, stats


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='L1 Pruning Demo for SegFormer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num_classes', type=int, default=21,
                        help='Number of output classes')
    parser.add_argument('--phi', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer model variant')
    parser.add_argument('--pruning_ratio', type=float, default=0.3,
                        help='Ratio of channels to prune (0.0-1.0)')
    parser.add_argument('--input_size', type=int, default=512,
                        help='Input image size (square)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save pruned model (default: auto-generated with timestamp)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Validate pruning ratio
    if not 0.0 <= args.pruning_ratio <= 1.0:
        print("Error: pruning_ratio must be between 0.0 and 1.0")
        return
    
    # Run demo
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    
    pruned_model, stats = demo_l1_pruning(
        num_classes=args.num_classes,
        phi=args.phi,
        pruning_ratio=args.pruning_ratio,
        input_shape=input_shape,
        device=device
    )
    
    # Optionally save the pruned model
    if args.save_path:
        save_path = args.save_path
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'pruned_segformer_{args.phi}_ratio{args.pruning_ratio:.2f}_{timestamp}.pth'
    
    print(f"\nSaving pruned model to: {save_path}")
    torch.save(pruned_model.state_dict(), save_path)
    print("✓ Model saved successfully!")


if __name__ == '__main__':
    main()
