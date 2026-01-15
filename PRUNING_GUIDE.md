# L1 Pruning Algorithm for SegFormer

This document provides a comprehensive guide to the L1 pruning implementation for the SegFormer semantic segmentation model.

## Table of Contents
- [Overview](#overview)
- [Algorithm Description](#algorithm-description)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Understanding the Results](#understanding-the-results)
- [Best Practices](#best-practices)

## Overview

L1 pruning is a model compression technique that reduces the size and computational cost of neural networks by removing less important channels. This implementation provides a global L1 norm-based pruning algorithm specifically designed for SegFormer models.

### Key Features

- **Global Pruning**: Evaluates channel importance across all layers simultaneously
- **L1 Norm Based**: Uses L1 norm as the importance metric
- **Configurable**: Adjustable pruning ratio from 0% to 100%
- **Validation**: Built-in forward pass validation
- **Statistics**: Detailed before/after comparison

## Algorithm Description

The L1 pruning algorithm follows these steps:

### 1. Channel Importance Calculation
For each Conv2d layer in the network:
```
For each output channel i:
    importance[i] = ||W_i||_1
```
Where `W_i` is the weight tensor for channel i, and `||·||_1` is the L1 norm (sum of absolute values).

### 2. Global Sorting
All channels from all layers are sorted together by their importance scores in ascending order (least important first).

### 3. Channel Selection
Based on the pruning ratio `r`:
```
num_prune = int(total_channels × r)
channels_to_prune = sorted_channels[:num_prune]
```

### 4. Pruning Application
Two approaches are supported:
- **Soft Pruning**: Zero out weights of pruned channels (demonstrated in this implementation)
- **Hard Pruning**: Physically remove channels (requires model-specific rebuilding logic)

### 5. Validation
The pruned model is validated with a forward pass to ensure it still functions correctly.

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- Other dependencies from `requirements.txt`

### Install Dependencies
```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Quick Start

Run the pruning demo with default settings:

```bash
python prune_demo.py
```

This will:
1. Create a SegFormer-b0 model with 21 classes
2. Apply 30% pruning ratio
3. Validate the pruned model
4. Save the pruned model as `pruned_segformer_b0_ratio0.30.pth`

## Usage Examples

### Example 1: Basic Pruning

```bash
python prune_demo.py --phi b0 --pruning_ratio 0.3 --num_classes 21
```

### Example 2: Aggressive Pruning

```bash
python prune_demo.py --phi b1 --pruning_ratio 0.5 --num_classes 21
```

### Example 3: Conservative Pruning

```bash
python prune_demo.py --phi b0 --pruning_ratio 0.1 --num_classes 8
```

### Example 4: Custom Input Size

```bash
python prune_demo.py --input_size 256 --batch_size 4
```

### Example 5: CPU-Only Execution

```bash
python prune_demo.py --device cpu
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_classes` | int | 21 | Number of segmentation classes |
| `--phi` | str | b0 | Model variant (b0-b5) |
| `--pruning_ratio` | float | 0.3 | Pruning ratio (0.0-1.0) |
| `--input_size` | int | 512 | Input image size |
| `--batch_size` | int | 1 | Batch size for validation |
| `--device` | str | auto | Device (auto/cuda/cpu) |

## Using the Pruning Module Programmatically

```python
import torch
from nets.segformer import SegFormer
from nets.pruning import L1Pruner, validate_pruned_model

# Create model
model = SegFormer(num_classes=21, phi='b0', pretrained=False)
model.eval()

# Create pruner
pruner = L1Pruner(model, pruning_ratio=0.3)

# Execute pruning
pruned_model = pruner.prune()

# Get statistics
stats = pruner.get_pruning_statistics()
print(f"Pruned {stats['pruned_channels']} channels")

# Validate
input_shape = (1, 3, 512, 512)
is_valid = validate_pruned_model(pruned_model, input_shape)

# Save pruned model
torch.save(pruned_model.state_dict(), 'my_pruned_model.pth')
```

## API Reference

### L1Pruner Class

#### `__init__(model, pruning_ratio=0.3)`
Initialize the pruner.

**Parameters:**
- `model` (nn.Module): Model to prune
- `pruning_ratio` (float): Ratio of channels to prune (0.0-1.0)

#### `compute_channel_importance()`
Compute L1 norm for each channel.

**Returns:** List of (layer_name, channel_idx, importance_score)

#### `global_sort_channels()`
Sort channels by importance globally.

**Returns:** Sorted list of (layer_name, channel_idx, importance_score)

#### `determine_pruning_masks()`
Determine which channels to prune.

**Returns:** Dictionary of layer names to boolean masks

#### `apply_pruning_masks()`
Apply soft pruning by zeroing weights.

#### `rebuild_model()`
Rebuild model with pruned channels removed (currently applies soft pruning).

**Returns:** Pruned model

#### `get_pruning_statistics()`
Get pruning statistics.

**Returns:** Dictionary with statistics

#### `prune()`
Execute complete pruning pipeline.

**Returns:** Pruned model

### Utility Functions

#### `validate_pruned_model(model, input_shape)`
Validate that pruned model works.

**Parameters:**
- `model` (nn.Module): Model to validate
- `input_shape` (tuple): Input tensor shape

**Returns:** Boolean indicating success

#### `count_parameters(model)`
Count model parameters.

**Parameters:**
- `model` (nn.Module): Model to analyze

**Returns:** Tuple of (total_params, trainable_params)

## Understanding the Results

### Output Interpretation

When you run the demo, you'll see output like:

```
============================================================
  Pruning Statistics
============================================================
  total_channels: 13260
  pruned_channels: 3978
  remaining_channels: 9282
  pruning_ratio_actual: 0.3000
  total_parameters: 4710656
```

**Key Metrics:**
- **total_channels**: Total Conv2d output channels in the model
- **pruned_channels**: Number of channels pruned
- **remaining_channels**: Channels kept
- **pruning_ratio_actual**: Actual achieved pruning ratio
- **total_parameters**: Total model parameters

### Soft vs Hard Pruning

This implementation uses **soft pruning** by default:
- Weights of pruned channels are zeroed
- Model structure remains unchanged
- Parameter count stays the same
- Computation can be reduced with sparse operations

For **hard pruning** (true parameter reduction):
- Requires rebuilding the model architecture
- Physically removes channels
- Reduces parameter count
- Model-specific implementation needed

## Best Practices

### 1. Start Conservative
Begin with a low pruning ratio (e.g., 0.1-0.2) and gradually increase:

```bash
python prune_demo.py --pruning_ratio 0.1
```

### 2. Fine-tune After Pruning
The pruned model typically needs fine-tuning to recover accuracy:

```python
# After pruning
pruned_model = pruner.prune()

# Fine-tune on your dataset
optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-5)
# ... training loop ...
```

### 3. Validate on Real Data
Test the pruned model on your actual validation set:

```python
# Load your data
val_loader = DataLoader(val_dataset, batch_size=1)

# Evaluate
pruned_model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = pruned_model(images)
        # Calculate metrics
```

### 4. Monitor Accuracy vs Compression
Track the trade-off between model size and performance:

| Pruning Ratio | Channels Pruned | Accuracy Change |
|---------------|-----------------|-----------------|
| 0.1 | ~10% | -0.5% |
| 0.2 | ~20% | -1.2% |
| 0.3 | ~30% | -2.5% |
| 0.5 | ~50% | -5.0% |

(Note: These are example values; actual results depend on your model and dataset)

### 5. Consider Layer-wise Pruning
For production use, consider implementing layer-specific pruning ratios:
- Prune early layers less (important for feature extraction)
- Prune later layers more (higher redundancy)

### 6. Combine with Other Techniques
L1 pruning can be combined with:
- Quantization (reduce precision)
- Knowledge distillation (transfer knowledge)
- Neural architecture search (find optimal architectures)

## Extending the Implementation

### Adding Hard Pruning

To implement hard pruning (true parameter removal), you need to:

1. Track input/output channel dependencies
2. Rebuild each layer with updated channel counts
3. Copy weights from original to pruned model
4. Handle batch normalization layers

Example skeleton:

```python
def hard_prune_conv_layer(conv_layer, out_mask, in_mask):
    """Rebuild a Conv2d layer with pruned channels."""
    new_out_channels = int(out_mask.sum())
    new_in_channels = int(in_mask.sum())
    
    new_conv = nn.Conv2d(
        new_in_channels,
        new_out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    
    # Copy pruned weights
    new_conv.weight.data = conv_layer.weight.data[out_mask][:, in_mask]
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[out_mask]
    
    return new_conv
```

### Custom Importance Metrics

You can modify the importance calculation:

```python
def compute_l2_importance(self):
    """Use L2 norm instead of L1."""
    importance_list = []
    for name, module in self.model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            for channel_idx in range(weight.shape[0]):
                channel_weight = weight[channel_idx]
                l2_norm = torch.sqrt(torch.sum(channel_weight ** 2)).item()
                importance_list.append((name, channel_idx, l2_norm))
    return importance_list
```

## Troubleshooting

### Issue: "Model validation failed"
**Solution:** Ensure input shape matches model expectations and all dependencies are installed.

### Issue: "Pruning ratio too aggressive"
**Solution:** Reduce pruning ratio or implement iterative pruning with fine-tuning between iterations.

### Issue: "Out of memory"
**Solution:** Reduce batch size or input size:
```bash
python prune_demo.py --input_size 256 --batch_size 1
```

## Citation

If you use this L1 pruning implementation in your research, please cite:

```bibtex
@misc{segformer_l1_pruning,
  title={L1 Pruning Implementation for SegFormer},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/xiaohuaguaidou/SegFormer_ice-1}}
}
```

## License

This implementation follows the same license as the SegFormer repository.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- SegFormer: [Original Paper](https://arxiv.org/abs/2105.15203)
- Neural Network Pruning: [Survey Paper](https://arxiv.org/abs/2102.00554)
