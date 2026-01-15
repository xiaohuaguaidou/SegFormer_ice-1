# SegFormer 剪枝功能使用文档

## 功能介绍

本项目为 SegFormer 网络实现了基于 L1 范数的全局剪枝功能，支持对网络进行结构化剪枝，以减少参数量和计算量。

### 主要特性

- **L1 范数全局剪枝**: 基于卷积层权重的 L1 范数进行通道剪枝
- **结构化剪枝**: 剪枝整个通道/过滤器，而非单个权重
- **保护关键层**: 自动忽略第一层和最后的分类层
- **详细统计**: 提供参数量、FLOPs、推理时间的对比分析
- **模型兼容性**: 剪枝后的模型可以直接用于训练和推理

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
    └── b2/
        ├── b2_pruned_0.5.pth          # 剪枝后的模型权重
        └── pruning_results.json        # 剪枝统计结果
```

### 统计结果说明

`pruning_results.json` 包含以下信息：

```json
{
    "model_config": {
        "phi": "b2",
        "num_classes": 8,
        "input_shape": [512, 512]
    },
    "pruning_config": {
        "method": "l1",
        "pruning_rate": 0.5
    },
    "original_model": {
        "parameters": 25000000,
        "parameters_M": 25.0,
        "flops": 50000000000,
        "flops_G": 50.0
    },
    "pruned_model": {
        "parameters": 12500000,
        "parameters_M": 12.5,
        "flops": 25000000000,
        "flops_G": 25.0
    },
    "reduction": {
        "parameters_reduction_percent": 50.0,
        "flops_reduction_percent": 50.0
    },
    "model_save_path": "logs_pruning/l1/b2/b2_pruned_0.5.pth"
}
```

## 剪枝后的微调

剪枝后的模型需要进行微调以恢复精度。可以使用原有的 `train.py` 加载剪枝后的模型进行微调：

```bash
python train.py \
    --model_path logs_pruning/l1/b2/b2_pruned_0.5.pth \
    --phi b2 \
    --num_classes 8 \
    --input_shape 512 512
```

### 微调建议

1. **学习率**: 使用较小的学习率（如原始学习率的 1/10）
2. **训练轮数**: 根据数据集大小，通常需要 50-100 个 epoch
3. **冻结策略**: 可以先冻结骨干网络，只训练分类头，然后解冻全部网络微调

## 预期效果

不同剪枝率的预期效果（以 50% 剪枝率为例）：

| 指标 | 变化 |
|-----|------|
| 参数量 | 减少约 40-50% |
| FLOPs | 减少约 40-50% |
| 推理速度 | 提升约 1.2-1.5x |
| 准确率 | 初始下降 5-10%，微调后可恢复到原始精度的 95-98% |

### 不同剪枝率建议

- **0.3 (30%)**: 保守策略，精度几乎无损失，速度略有提升
- **0.5 (50%)**: 平衡策略，速度和精度的良好权衡
- **0.7 (70%)**: 激进策略，显著减少参数，需要更多微调

## 注意事项

1. **剪枝率选择**
   - 建议从 0.3-0.5 开始尝试
   - 过高的剪枝率可能导致精度难以恢复
   - 不同的骨干网络对剪枝的敏感度不同

2. **模型加载**
   - 确保剪枝前的模型已经训练好
   - 剪枝一个未训练的模型效果不佳

3. **设备要求**
   - 剪枝过程需要一定的显存（建议至少 6GB）
   - 可以在 CPU 上运行，但速度较慢

4. **版本兼容性**
   - 确保 `torch-pruning` 库版本 >= 1.3.0
   - PyTorch 版本建议 >= 1.10.0

5. **微调必要性**
   - 剪枝后的模型精度会有所下降
   - 必须进行微调以恢复精度
   - 微调时间通常比初始训练短

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

### 问题 3: 剪枝后精度下降严重

**解决方案**:
- 降低剪枝率
- 增加微调的训练轮数
- 使用更大的学习率进行微调
- 确保原始模型已经充分训练

### 问题 4: 剪枝速度很慢

**解决方案**:
- 使用 CUDA 加速
- 减小 `input_shape`
- 使用较小的骨干网络版本

## 技术细节

### 剪枝原理

L1 范数剪枝的工作原理：

1. **重要性评估**: 计算每个卷积通道的 L1 范数作为重要性分数
2. **全局排序**: 对所有可剪枝层的通道进行全局排序
3. **通道剪除**: 移除重要性最低的通道
4. **依赖更新**: 自动更新相关层的连接关系

### 保护的层

以下类型的层会被自动保护，不参与剪枝：

- 第一个卷积层（网络输入层）
- 最后的分类层（输出层）
- 显式指定的 `ignored_layers`

### 全局 vs 局部剪枝

- **全局剪枝** (本实现): 统一对整个网络应用剪枝率，根据全局重要性排序
- **局部剪枝**: 对每一层单独应用剪枝率

全局剪枝通常能获得更好的效果。

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
