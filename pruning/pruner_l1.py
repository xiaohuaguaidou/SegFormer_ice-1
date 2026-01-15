# -*- coding: utf-8 -*-
"""
L1范数剪枝器
基于权重L1范数进行全局结构化剪枝
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .pruner_base import BasePruner

try:
    import torch_pruning as tp
except ImportError:
    print("警告: torch_pruning 未安装，请运行: pip install torch-pruning")
    tp = None


class L1Pruner(BasePruner):
    """
    L1范数剪枝器
    使用L1范数作为通道重要性度量，进行全局结构化剪枝
    """
    
    def __init__(self, model: nn.Module, pruning_rate: float):
        """
        初始化L1剪枝器
        
        Args:
            model: 待剪枝的模型
            pruning_rate: 剪枝率，范围[0, 1)
        """
        super().__init__(model, pruning_rate)
        
        if tp is None:
            raise ImportError(
                "torch_pruning 未安装。请运行: pip install torch-pruning"
            )
    
    def prune(self, example_inputs: torch.Tensor, 
              ignored_layers: Optional[List[nn.Module]] = None,
              **kwargs) -> nn.Module:
        """
        执行L1范数全局剪枝
        
        Args:
            example_inputs: 示例输入，用于追踪模型依赖关系
            ignored_layers: 需要忽略的层列表（如第一层和最后的分类层）
            **kwargs: 其他参数
            
        Returns:
            剪枝后的模型
        """
        print(f"\n开始执行L1范数全局剪枝 (剪枝率: {self.pruning_rate:.2%})...")
        
        # 将模型设置为评估模式
        self.original_model.eval()
        
        # 确保example_inputs在正确的设备上
        device = next(self.original_model.parameters()).device
        example_inputs = example_inputs.to(device)
        
        # 自动收集需要忽略的层
        if ignored_layers is None:
            ignored_layers = self._collect_ignored_layers()
        
        print(f"忽略的层数量: {len(ignored_layers)}")
        
        try:
            # 创建重要性评估器 - 使用L1范数
            importance = tp.importance.MagnitudeImportance(p=1)  # p=1 表示L1范数
            
            # 创建剪枝器
            pruner = tp.pruner.MagnitudePruner(
                model=self.original_model,
                example_inputs=example_inputs,
                importance=importance,
                pruning_ratio=self.pruning_rate,
                ignored_layers=ignored_layers,
                global_pruning=True,  # 全局剪枝
            )
            
            # 执行剪枝
            print("正在分析模型结构...")
            pruner.step()
            
            print("剪枝完成！")
            
            # 保存剪枝后的模型
            self.pruned_model = self.original_model
            
            # 记录剪枝信息
            self._record_pruning_info()
            
            return self.pruned_model
            
        except Exception as e:
            print(f"剪枝过程中出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _collect_ignored_layers(self) -> List[nn.Module]:
        """
        自动收集需要忽略的层
        通常包括：
        1. 第一个卷积层
        2. 最后的分类层
        3. 批归一化层
        
        Returns:
            需要忽略的层列表
        """
        ignored_layers = []
        
        # 收集所有卷积层
        conv_layers = []
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if len(conv_layers) > 0:
            # 忽略第一个卷积层（特征提取的入口）
            first_conv_name, first_conv = conv_layers[0]
            ignored_layers.append(first_conv)
            print(f"忽略第一个卷积层: {first_conv_name}")
        
        # 查找最后的分类层（通常是1x1卷积）
        for name, module in self.original_model.named_modules():
            # SegFormer的分类层通常命名为 linear_pred 或类似名称
            if 'linear_pred' in name or 'classifier' in name or 'head' in name:
                if isinstance(module, nn.Conv2d):
                    ignored_layers.append(module)
                    print(f"忽略分类层: {name}")
        
        return ignored_layers
    
    def _record_pruning_info(self):
        """
        记录剪枝信息
        """
        from .utils_pruning import count_parameters
        
        if self.pruned_model is not None:
            pruned_params = count_parameters(self.pruned_model)
            original_params = count_parameters(self.original_model)
            
            self.pruning_info = {
                'pruning_method': 'L1',
                'pruning_rate': self.pruning_rate,
                'original_parameters': original_params,
                'pruned_parameters': pruned_params,
                'parameter_reduction_rate': (1 - pruned_params / original_params) * 100
            }
