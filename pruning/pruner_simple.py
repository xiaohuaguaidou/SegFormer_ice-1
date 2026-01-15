# -*- coding: utf-8 -*-
"""
简化的L1范数剪枝器
针对SegFormer的特殊结构（包含自定义KAN层）提供更简单的剪枝方案
"""

import torch
import torch.nn as nn
import copy
from typing import List, Optional
from .pruner_base import BasePruner


class SimpleL1Pruner(BasePruner):
    """
    简化的L1范数剪枝器
    专门针对包含自定义模块的SegFormer模型
    采用简化的剪枝策略，只对backbone中的标准卷积层进行剪枝
    """
    
    def __init__(self, model: nn.Module, pruning_rate: float):
        """
        初始化简化L1剪枝器
        
        Args:
            model: 待剪枝的模型
            pruning_rate: 剪枝率，范围[0, 1)
        """
        super().__init__(model, pruning_rate)
    
    def prune(self, example_inputs: torch.Tensor, 
              ignored_layers: Optional[List[nn.Module]] = None,
              **kwargs) -> nn.Module:
        """
        执行简化的L1范数剪枝
        
        此方法不实际修改模型结构，而是生成一个剪枝计划
        用户可以根据此计划手动调整模型或进行知识蒸馏
        
        Args:
            example_inputs: 示例输入
            ignored_layers: 需要忽略的层列表
            **kwargs: 其他参数
            
        Returns:
            原始模型（带有剪枝标记，但结构未改变）
        """
        print(f"\n开始执行简化L1范数剪枝分析 (剪枝率: {self.pruning_rate:.2%})...")
        print("注意: 由于SegFormer使用了自定义KAN层，此实现生成剪枝计划而不直接修改模型")
        
        # 将模型设置为评估模式
        self.original_model.eval()
        
        # 确保example_inputs在正确的设备上
        device = next(self.original_model.parameters()).device
        example_inputs = example_inputs.to(device)
        
        # 分析可剪枝的卷积层
        prunable_convs = self._collect_prunable_layers()
        print(f"\n发现 {len(prunable_convs)} 个可剪枝的卷积层")
        
        # 计算每个卷积层的L1重要性
        importances = self._calculate_importance(prunable_convs)
        
        # 生成剪枝计划
        pruning_plan = self._generate_pruning_plan(prunable_convs, importances)
        
        # 打印剪枝计划
        self._print_pruning_plan(pruning_plan)
        
        # 由于无法直接修改包含自定义模块的模型，返回原始模型
        # 并在pruning_info中保存剪枝计划
        self.pruned_model = self.original_model
        self.pruning_info['pruning_plan'] = pruning_plan
        self.pruning_info['note'] = (
            "由于模型包含自定义KAN层，torch_pruning无法直接应用。"
            "建议使用以下方法之一："
            "\n1. 基于此剪枝计划进行知识蒸馏"
            "\n2. 手动调整模型结构"
            "\n3. 使用标准SegFormer（无KAN层）进行剪枝"
        )
        
        # 记录基本信息
        self._record_pruning_info()
        
        return self.pruned_model
    
    def _collect_prunable_layers(self) -> List[tuple]:
        """
        收集可以剪枝的层
        只收集backbone中的标准卷积层
        
        Returns:
            (name, module) 元组的列表
        """
        prunable_layers = []
        
        if hasattr(self.original_model, 'backbone'):
            for name, module in self.original_model.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    # 跳过关键的结构层
                    # 使用更健壮的检查方式
                    is_critical_layer = False
                    
                    # Check if this is a patch embedding layer
                    if 'patch_embed' in name.lower() or 'proj' in name.lower():
                        is_critical_layer = True
                    
                    # Check if this is the first conv in a block
                    parent_modules = name.split('.')
                    if len(parent_modules) > 0 and parent_modules[0] in ['conv1', 'downsample']:
                        is_critical_layer = True
                    
                    if not is_critical_layer:
                        full_name = f"backbone.{name}"
                        prunable_layers.append((full_name, module))
        
        return prunable_layers
    
    def _calculate_importance(self, prunable_convs: List[tuple]) -> dict:
        """
        计算每个卷积层输出通道的L1范数重要性
        
        Args:
            prunable_convs: 可剪枝的卷积层列表
            
        Returns:
            字典: {layer_name: importance_scores}
        """
        importances = {}
        
        for name, module in prunable_convs:
            # 获取卷积核权重
            weight = module.weight.data  # shape: [out_channels, in_channels, k, k]
            
            # 计算每个输出通道的L1范数
            # 对每个输出通道，计算所有权重的L1范数
            channel_importance = weight.abs().sum(dim=[1, 2, 3])  # shape: [out_channels]
            
            importances[name] = channel_importance.cpu()
        
        return importances
    
    def _generate_pruning_plan(self, prunable_convs: List[tuple], 
                               importances: dict) -> dict:
        """
        生成剪枝计划
        
        Args:
            prunable_convs: 可剪枝的卷积层列表
            importances: 重要性分数
            
        Returns:
            剪枝计划字典
        """
        plan = {}
        
        for name, module in prunable_convs:
            importance = importances[name]
            out_channels = module.out_channels
            
            # 计算要剪枝的通道数
            num_prune = int(out_channels * self.pruning_rate)
            num_keep = out_channels - num_prune
            
            if num_keep < 1:
                num_keep = 1
                num_prune = out_channels - 1
            
            # 找出要保留的通道（重要性最高的）
            _, indices = importance.sort(descending=True)
            keep_indices = indices[:num_keep].sort()[0].tolist()
            prune_indices = indices[num_keep:].sort()[0].tolist()
            
            plan[name] = {
                'total_channels': out_channels,
                'keep_channels': num_keep,
                'prune_channels': num_prune,
                'keep_indices': keep_indices,
                'prune_indices': prune_indices,
                'reduction_rate': (num_prune / out_channels) * 100
            }
        
        return plan
    
    def _print_pruning_plan(self, pruning_plan: dict):
        """
        打印剪枝计划
        
        Args:
            pruning_plan: 剪枝计划字典
        """
        print(f"\n{'='*80}")
        print("剪枝计划:")
        print(f"{'='*80}")
        
        total_original = 0
        total_pruned = 0
        
        for layer_name, plan in pruning_plan.items():
            total_original += plan['total_channels']
            total_pruned += plan['prune_channels']
            
            print(f"\n层: {layer_name}")
            print(f"  原始通道数: {plan['total_channels']}")
            print(f"  保留通道数: {plan['keep_channels']}")
            print(f"  剪枝通道数: {plan['prune_channels']}")
            print(f"  减少率: {plan['reduction_rate']:.2f}%")
        
        print(f"\n{'='*80}")
        print(f"总计:")
        print(f"  原始总通道数: {total_original}")
        print(f"  剪枝总通道数: {total_pruned}")
        print(f"  平均减少率: {(total_pruned/total_original)*100:.2f}%")
        print(f"{'='*80}\n")
    
    def _record_pruning_info(self):
        """
        记录剪枝信息
        """
        from .utils_pruning import count_parameters
        
        params = count_parameters(self.original_model)
        
        self.pruning_info.update({
            'pruning_method': 'SimpleL1',
            'pruning_rate': self.pruning_rate,
            'original_parameters': params,
            'pruned_parameters': params,  # 结构未改变
            'parameter_reduction_rate': 0,  # 实际未减少
            'status': 'plan_only'
        })
