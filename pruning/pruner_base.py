# -*- coding: utf-8 -*-
"""
剪枝器基类
定义剪枝器的通用接口
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict


class BasePruner(ABC):
    """
    剪枝器基类
    所有剪枝器都应该继承此类并实现prune方法
    """
    
    def __init__(self, model: nn.Module, pruning_rate: float):
        """
        初始化剪枝器
        
        Args:
            model: 待剪枝的模型
            pruning_rate: 剪枝率，范围[0, 1)
        """
        if not 0 <= pruning_rate < 1:
            raise ValueError(f"剪枝率必须在[0, 1)范围内，当前值: {pruning_rate}")
        
        self.original_model = model
        self.pruned_model = None
        self.pruning_rate = pruning_rate
        self.pruning_info = {}
    
    @abstractmethod
    def prune(self, example_inputs: torch.Tensor, **kwargs) -> nn.Module:
        """
        执行剪枝操作
        
        Args:
            example_inputs: 示例输入，用于追踪模型结构
            **kwargs: 其他剪枝参数
            
        Returns:
            剪枝后的模型
        """
        pass
    
    def get_pruned_model(self) -> nn.Module:
        """
        获取剪枝后的模型
        
        Returns:
            剪枝后的模型，如果尚未剪枝则返回None
        """
        if self.pruned_model is None:
            raise RuntimeError("模型尚未剪枝，请先调用prune()方法")
        return self.pruned_model
    
    def save_pruned_model(self, save_path: str):
        """
        保存剪枝后的模型
        
        Args:
            save_path: 保存路径
        """
        if self.pruned_model is None:
            raise RuntimeError("模型尚未剪枝，无法保存")
        
        try:
            torch.save(self.pruned_model.state_dict(), save_path)
            print(f"剪枝模型已保存到: {save_path}")
        except Exception as e:
            print(f"保存剪枝模型时出错: {e}")
            raise
    
    @staticmethod
    def load_pruned_model(model: nn.Module, load_path: str, 
                         device: torch.device = None) -> nn.Module:
        """
        加载剪枝后的模型权重
        
        Args:
            model: 目标模型（应该是剪枝后的结构）
            load_path: 权重文件路径
            device: 加载到的设备
            
        Returns:
            加载权重后的模型
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"成功从 {load_path} 加载剪枝模型权重")
            return model
        except Exception as e:
            print(f"加载剪枝模型权重时出错: {e}")
            raise
    
    def get_pruning_info(self) -> Dict[str, Any]:
        """
        获取剪枝信息
        
        Returns:
            剪枝信息字典
        """
        return self.pruning_info
