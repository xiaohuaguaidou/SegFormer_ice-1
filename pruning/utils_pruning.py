# -*- coding: utf-8 -*-
"""
剪枝工具函数
提供模型统计、保存加载等实用工具
"""

import json
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def count_parameters(model: nn.Module) -> int:
    """
    统计模型的总参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        总参数量
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    统计模型的可训练参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        可训练参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model: nn.Module, input_shape: Tuple[int, int, int, int], 
                          device: torch.device, num_iterations: int = 100) -> float:
    """
    测量模型的推理时间
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (batch_size, channels, height, width)
        device: 运行设备
        num_iterations: 测试迭代次数
        
    Returns:
        平均推理时间（秒）
    """
    model.eval()
    model.to(device)
    
    # 生成随机输入
    dummy_input = torch.randn(input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 测量时间
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    return avg_time


def calculate_flops(model: nn.Module, input_shape: Tuple[int, int, int], 
                    device: torch.device = None) -> Tuple[float, int]:
    """
    计算模型的FLOPs和参数量
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (channels, height, width)
        device: 运行设备
        
    Returns:
        (FLOPs, 参数量)
    """
    try:
        from ptflops import get_model_complexity_info
        
        if device is None:
            # Safely get device from model parameters
            try:
                device = next(model.parameters()).device
            except StopIteration:
                # Model has no parameters, use CPU as default
                device = torch.device('cpu')
        
        model_copy = model.to(device)
        
        macs, params = get_model_complexity_info(
            model_copy, 
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        
        # MACs to FLOPs (通常 FLOPs = 2 * MACs)
        flops = 2 * macs
        
        return flops, params
    except Exception as e:
        print(f"计算FLOPs时出错: {e}")
        return 0, count_parameters(model)


def save_pruning_results(results: Dict[str, Any], save_path: str):
    """
    保存剪枝结果到JSON文件
    
    Args:
        results: 剪枝结果字典
        save_path: 保存路径
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"剪枝结果已保存到: {save_path}")
    except Exception as e:
        print(f"保存剪枝结果时出错: {e}")


def load_pruning_results(load_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载剪枝结果
    
    Args:
        load_path: 加载路径
        
    Returns:
        剪枝结果字典
    """
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"加载剪枝结果时出错: {e}")
        return {}


def print_model_info(model: nn.Module, model_name: str = "Model", 
                    input_shape: Tuple[int, int, int] = None,
                    device: torch.device = None):
    """
    打印模型的详细信息
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
        input_shape: 输入形状 (channels, height, width)
        device: 运行设备
    """
    print(f"\n{'='*60}")
    print(f"{model_name} 信息:")
    print(f"{'='*60}")
    
    # 参数量
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # FLOPs
    if input_shape is not None:
        flops, _ = calculate_flops(model, input_shape, device)
        if flops > 0:
            print(f"FLOPs: {flops:,} ({flops/1e9:.2f}G)")
    
    # 推理时间
    if input_shape is not None and device is not None:
        inference_time = measure_inference_time(
            model, 
            (1, input_shape[0], input_shape[1], input_shape[2]), 
            device,
            num_iterations=50
        )
        print(f"平均推理时间: {inference_time*1000:.2f} ms")
    
    print(f"{'='*60}\n")


def compare_models(original_model: nn.Module, pruned_model: nn.Module,
                  input_shape: Tuple[int, int, int] = None,
                  device: torch.device = None):
    """
    对比原始模型和剪枝后模型的统计信息
    
    Args:
        original_model: 原始模型
        pruned_model: 剪枝后模型
        input_shape: 输入形状 (channels, height, width)
        device: 运行设备
    """
    print(f"\n{'='*80}")
    print(f"模型对比分析")
    print(f"{'='*80}")
    
    # 参数量对比
    orig_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    params_reduction = (1 - pruned_params / orig_params) * 100
    
    print(f"\n参数量对比:")
    print(f"  原始模型: {orig_params:,} ({orig_params/1e6:.2f}M)")
    print(f"  剪枝模型: {pruned_params:,} ({pruned_params/1e6:.2f}M)")
    print(f"  减少: {params_reduction:.2f}%")
    
    # FLOPs对比
    if input_shape is not None:
        orig_flops, _ = calculate_flops(original_model, input_shape, device)
        pruned_flops, _ = calculate_flops(pruned_model, input_shape, device)
        
        if orig_flops > 0 and pruned_flops > 0:
            flops_reduction = (1 - pruned_flops / orig_flops) * 100
            print(f"\nFLOPs对比:")
            print(f"  原始模型: {orig_flops:,} ({orig_flops/1e9:.2f}G)")
            print(f"  剪枝模型: {pruned_flops:,} ({pruned_flops/1e9:.2f}G)")
            print(f"  减少: {flops_reduction:.2f}%")
    
    # 推理时间对比
    if input_shape is not None and device is not None:
        try:
            orig_time = measure_inference_time(
                original_model,
                (1, input_shape[0], input_shape[1], input_shape[2]),
                device,
                num_iterations=50
            )
            pruned_time = measure_inference_time(
                pruned_model,
                (1, input_shape[0], input_shape[1], input_shape[2]),
                device,
                num_iterations=50
            )
            speedup = orig_time / pruned_time
            time_reduction = (1 - pruned_time / orig_time) * 100
            
            print(f"\n推理时间对比:")
            print(f"  原始模型: {orig_time*1000:.2f} ms")
            print(f"  剪枝模型: {pruned_time*1000:.2f} ms")
            print(f"  加速: {speedup:.2f}x")
            print(f"  减少: {time_reduction:.2f}%")
        except Exception as e:
            print(f"\n推理时间测量失败: {e}")
    
    print(f"\n{'='*80}\n")
