# SegFormer 剪枝功能使用文档

## 功能介绍

本项目为 SegFormer 网络实现了基于 L1 范数的全局剪枝功能，支持对网络进行结构化剪枝分析，以规划参数量和计算量的减少。

**重要说明**: 由于本项目的 SegFormer 实现使用了自定义 KAN (Kolmogorov-Arnold Network) 层，标准的 torch-pruning 库无法直接应用结构化剪枝。因此，本实现提供了以下功能：

1. **标准 L1 剪枝器**: 尝试使用 torch-pruning 进行自动剪枝（适用于标准 SegFormer）
2. **简化 L1 剪枝器**: 当自动剪枝失败时，生成详细的剪枝计划，用于指导知识蒸馏或手动优化

### 主要特性

- **L1 范数分析**: 基于卷积层权重的 L1 范数评估通道重要性
- **剪枝计划生成**: 生成详细的每层剪枝建议
- **保护关键层**: 自动忽略第一层、patch embedding 和分类层
- **详细统计**: 提供参数量、FLOPs、推理时间的对比分析
- **自动回退**: 当直接剪枝失败时，自动切换到剪枝计划模式

## 安装依赖

### 安装 torch-pruning

```bash
pip install torch-pruning
```

### 安装所有依赖

```bash
pip install -r requirements_pruning.txt
```

## 使用方法

### 基本用法

对预训练模型执行 50% 剪枝率的 L1 剪枝：

```bash
python prune_train.py \
    --model_path logs/your_model.pth \
    --pruning_method l1 \
    --pruning_rate 0.5 \
    --num_classes 8 \
    --phi b2 \
    --cuda
```

### 完整参数说明

#### 模型参数

- `--model_path`: 预训练模型路径（可选，默认为空表示随机初始化）
- `--num_classes`: 分类类别数（默认: 8）
- `--phi`: 骨干网络版本，可选 b0-b5（默认: b2）
- `--input_shape`: 输入图像尺寸 [height width]（默认: 512 512）

#### 剪枝参数

- `--pruning_method`: 剪枝方法，目前支持 l1（默认: l1）
- `--pruning_rate`: 剪枝率，范围 [0, 1)（默认: 0.5）

#### 运行参数

- `--cuda`: 是否使用 CUDA（添加此参数表示使用 CUDA）
- `--seed`: 随机种子（默认: 11）
- `--save_dir`: 剪枝模型保存目录（默认: logs_pruning）

### 使用示例

#### 1. 对预训练模型进行剪枝

```bash
python prune_train.py \
    --model_path logs/best_model.pth \
    --pruning_method l1 \
    --pruning_rate 0.5 \
    --num_classes 8 \
    --phi b2 \
    --input_shape 512 512 \
    --cuda
```

#### 2. 使用不同的剪枝率

30% 剪枝率（较保守）：
```bash
python prune_train.py \
    --model_path logs/best_model.pth \
    --pruning_rate 0.3 \
    --phi b2 \
    --cuda
```

70% 剪枝率（更激进）：
```bash
python prune_train.py \
    --model_path logs/best_model.pth \
    --pruning_rate 0.7 \
    --phi b2 \
    --cuda
```

#### 3. 对不同版本的骨干网络进行剪枝

SegFormer-B0:
```bash
python prune_train.py \
    --model_path logs/b0_model.pth \
    --phi b0 \
    --pruning_rate 0.5 \
    --cuda
```

SegFormer-B5:
```bash
python prune_train.py \
    --model_path logs/b5_model.pth \
    --phi b5 \
    --pruning_rate 0.4 \
    --cuda
```

## 输出结果

### 保存的文件

剪枝完成后，会在指定目录下生成以下文件：

```
logs_pruning/
└── l1/
    └── b0/
        ├── b0_pruned_0.3.pth          # 模型权重（结构未改变，但包含分析结果）
        └── pruning_results.json        # 剪枝统计结果和剪枝计划
```

### 统计结果说明

`pruning_results.json` 包含以下信息：

```json
{
    "model_config": {
        "phi": "b0",
        "num_classes": 8,
        "input_shape": [128, 128]
    },
    "pruning_config": {
        "method": "l1",
        "pruning_rate": 0.3,
        "actual_pruning_applied": false  // false表示生成了剪枝计划而非直接剪枝
    },
    "original_model": {
        "parameters": 4710656,
        "parameters_M": 4.71,
        "flops": 999362336,
        "flops_G": 1.0
    },
    "pruned_model": {
        "parameters": 4710656,  // 与原始相同（结构未改变）
        "parameters_M": 4.71,
        "flops": 999362336,
        "flops_G": 1.0
    },
    "reduction": {
        "parameters_reduction_percent": 0,  // 实际未减少
        "flops_reduction_percent": 0
    },
    "model_save_path": "logs_pruning/l1/b0/b0_pruned_0.3.pth",
    "pruning_info": {
        "pruning_plan": {
            "backbone.block1.0.attn.original_attn.sr": {
                "total_channels": 32,
                "keep_channels": 23,
                "prune_channels": 9,
                "keep_indices": [0, 1, 2, ...],  // 建议保留的通道索引
                "prune_indices": [7, 8, 10, ...],  // 建议剪枝的通道索引
                "reduction_rate": 28.12
            },
            // 更多层的剪枝计划...
        },
        "note": "由于模型包含自定义KAN层，torch_pruning无法直接应用..."
    }
}
```

### 如何使用剪枝计划

生成的剪枝计划可以用于以下场景：

#### 1. 知识蒸馏

使用剪枝计划中的通道数配置，创建一个更小的学生模型，然后使用知识蒸馏训练：

```python
# 根据剪枝计划创建更小的模型
student_model = create_pruned_segformer(pruning_plan)

# 使用知识蒸馏训练
distill_train(teacher_model=original_model, 
              student_model=student_model,
              ...)
```

#### 2. 手动模型优化

根据剪枝计划，手动调整模型结构，减少通道数。

#### 3. 重要性分析

分析哪些层对模型性能最重要，指导模型设计和优化。

## 剪枝后的微调

**注意**: 由于当前模型使用自定义 KAN 层，直接的结构化剪枝无法应用。以下是几种替代方案：

### 方案 1: 使用剪枝计划进行知识蒸馏

根据生成的剪枝计划，创建一个通道数减少的学生模型，然后使用知识蒸馏：

```python
# 1. 根据剪枝计划创建学生模型
student_model = create_smaller_segformer(pruning_plan)

# 2. 进行知识蒸馏训练
# 使用原始模型作为教师，训练学生模型
```

### 方案 2: 修改模型使用标准层

如果需要直接的结构化剪枝，可以将模型中的 KAN 层替换为标准的 MLP 层：

```python
# 修改 nets/segformer.py 中的 LightweightKANMLP
# 替换为标准的 MLP 实现
```

### 方案 3: 使用标准 SegFormer

使用不包含 KAN 层的标准 SegFormer 实现，可以直接应用 torch-pruning。

## 预期效果

使用简化剪枝分析器生成的剪枝计划：

| 指标 | 说明 |
|-----|------|
| 剪枝计划生成 | 成功，包含每层的详细通道剪枝建议 |
| 通道减少率 | 约等于设定的剪枝率（如 30% 剪枝率 → 约 30% 通道减少） |
| 实际参数减少 | 需要根据计划手动实现或使用知识蒸馏 |
| 推理速度提升 | 取决于基于计划创建的新模型 |

### 使用剪枝计划的预期收益（理论值）

基于 30% 剪枝率的剪枝计划，如果完全实施：

- **参数量**: 减少约 25-35%
- **FLOPs**: 减少约 20-30%
- **推理速度**: 提升约 1.2-1.3x
- **准确率**: 通过知识蒸馏，可维持原始精度的 95-98%

## 注意事项

1. **自定义模块限制**
   - 当前 SegFormer 实现使用了自定义 KAN 层
   - torch-pruning 无法自动处理这些自定义模块
   - 工具会自动生成剪枝计划而非直接修改模型

2. **剪枝计划的使用**
   - 查看 `pruning_results.json` 获取详细的剪枝建议
   - 使用剪枝计划指导知识蒸馏
   - 或根据计划手动创建更小的模型

3. **设备要求**
   - 分析过程需要一定的显存（建议至少 6GB）
   - 可以在 CPU 上运行，但速度较慢

4. **版本兼容性**
   - `torch-pruning` 库版本 >= 1.3.0
   - PyTorch 版本建议 >= 1.10.0

5. **剪枝率选择**
   - 建议从 0.2-0.4 开始尝试
   - 过高的剪枝率可能导致重要特征丢失

## 故障排除

### 问题 1: ImportError: torch_pruning 未安装

**解决方案**:
```bash
pip install torch-pruning
```

### 问题 2: CUDA out of memory

**解决方案**:
- 减小 `input_shape`
- 不使用 CUDA，在 CPU 上运行（去掉 `--cuda` 参数）
- 使用较小的骨干网络版本（如 b0 或 b1）

### 问题 3: 自动剪枝失败，切换到简化模式

这是**正常行为**。由于模型使用了自定义 KAN 层，torch-pruning 无法直接应用。工具会自动：
- 生成详细的剪枝计划
- 保存剪枝分析结果
- 提供后续使用建议

**建议的后续步骤**:
1. 查看 `pruning_results.json` 中的剪枝计划
2. 使用该计划进行知识蒸馏
3. 或考虑修改模型结构使用标准层

### 问题 4: 剪枝速度很慢

**解决方案**:
- 使用 CUDA 加速
- 减小 `input_shape`
- 使用较小的骨干网络版本

## 技术细节

### 剪枝原理

L1 范数剪枝的工作原理：

1. **重要性评估**: 计算每个卷积通道的 L1 范数作为重要性分数
2. **全局排序**: 对所有可剪枝层的通道进行评估
3. **生成计划**: 确定每层应该保留和剪除的通道
4. **输出建议**: 提供详细的剪枝计划供后续使用

### 保护的层

以下类型的层会被自动保护，不参与剪枝：

- 第一个卷积层（网络输入层）
- Patch embedding 层（下采样关键层）
- 整个 decode_head 模块（包含自定义 KAN 层）
- 所有 BatchNorm/LayerNorm 层

### 当前实现的限制

由于 SegFormer 使用了自定义 KAN 层，本实现提供两种模式：

1. **标准模式** (L1Pruner): 尝试使用 torch-pruning 自动剪枝
   - 仅适用于标准 PyTorch 模块
   - 对于包含自定义层的模型会失败

2. **简化模式** (SimpleL1Pruner): 生成剪枝计划
   - 分析每层的通道重要性
   - 生成详细的剪枝建议
   - 不直接修改模型结构
   - 可用于指导知识蒸馏或手动优化

工具会自动在两种模式间切换，优先尝试标准模式，失败时回退到简化模式。

## 进阶使用

### 自定义忽略层

如果需要自定义需要保护的层，可以修改 `pruner_l1.py` 中的 `_collect_ignored_layers` 方法。

### 集成到训练流程

可以在训练过程中周期性地进行剪枝，实现迭代剪枝：

```python
from pruning import L1Pruner

# 训练一定轮数后
pruner = L1Pruner(model, pruning_rate=0.1)
model = pruner.prune(example_inputs)

# 继续训练
# ...
```

## 参考文献

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [Torch-Pruning Documentation](https://github.com/VainF/Torch-Pruning)

## 许可证

本剪枝功能遵循与主项目相同的许可证。
