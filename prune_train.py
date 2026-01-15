# -*- coding: utf-8 -*-
"""
SegFormer模型剪枝脚本
支持L1范数全局剪枝
"""

import os
import argparse
import torch
import numpy as np
from nets.segformer import SegFormer
from pruning import L1Pruner, SimpleL1Pruner
from pruning.utils_pruning import (
    print_model_info,
    compare_models,
    save_pruning_results,
    count_parameters,
    calculate_flops
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SegFormer模型剪枝工具')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='',
                       help='预训练模型路径 (默认为空，将创建随机初始化的模型)')
    parser.add_argument('--num_classes', type=int, default=8,
                       help='分类类别数 (默认: 8)')
    parser.add_argument('--phi', type=str, default='b2',
                       choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                       help='骨干网络版本 (默认: b2)')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[512, 512],
                       help='输入图像尺寸 [height, width] (默认: 512 512)')
    
    # 剪枝参数
    parser.add_argument('--pruning_method', type=str, default='l1',
                       choices=['l1'],
                       help='剪枝方法 (默认: l1)')
    parser.add_argument('--pruning_rate', type=float, default=0.5,
                       help='剪枝率，范围[0, 1) (默认: 0.5)')
    
    # 运行参数
    parser.add_argument('--cuda', action='store_true', default=False,
                       help='是否使用CUDA')
    parser.add_argument('--seed', type=int, default=11,
                       help='随机种子 (默认: 11)')
    parser.add_argument('--save_dir', type=str, default='logs_pruning',
                       help='剪枝模型保存目录 (默认: logs_pruning)')
    
    return parser.parse_args()


def seed_everything(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path, num_classes, phi, device):
    """
    加载模型
    
    Args:
        model_path: 模型权重路径
        num_classes: 类别数
        phi: 骨干网络版本
        device: 运行设备
        
    Returns:
        加载的模型
    """
    print(f"\n{'='*60}")
    print("正在加载模型...")
    print(f"{'='*60}")
    
    # 创建模型
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False)
    
    # 加载预训练权重
    if model_path and os.path.exists(model_path):
        print(f"从 {model_path} 加载预训练权重...")
        try:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)
            
            # 过滤掉不匹配的键
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            
            print(f"成功加载 {len(load_key)} 个权重参数")
            if len(no_load_key) > 0:
                print(f"警告: {len(no_load_key)} 个权重参数未能加载")
        except Exception as e:
            print(f"加载权重时出错: {e}")
            print("将使用随机初始化的模型")
    else:
        if model_path:
            print(f"警告: 模型路径 {model_path} 不存在")
        print("使用随机初始化的模型")
    
    model.to(device)
    model.eval()
    
    return model


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    if args.cuda and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
    
    # 加载原始模型
    original_model = load_model(
        args.model_path,
        args.num_classes,
        args.phi,
        device
    )
    
    # 打印原始模型信息
    input_shape = (3, args.input_shape[0], args.input_shape[1])
    print_model_info(
        original_model,
        model_name="原始模型",
        input_shape=input_shape,
        device=device
    )
    
    # 创建示例输入
    example_inputs = torch.randn(1, 3, args.input_shape[0], args.input_shape[1]).to(device)
    
    # 执行剪枝
    print(f"\n{'='*60}")
    print(f"开始剪枝...")
    print(f"剪枝方法: {args.pruning_method.upper()}")
    print(f"剪枝率: {args.pruning_rate:.2%}")
    print(f"{'='*60}\n")
    
    # 尝试使用标准L1剪枝器，如果失败则使用简化版本
    pruned_model = None
    pruning_success = False
    
    if args.pruning_method == 'l1':
        try:
            print("尝试使用标准L1剪枝器（基于torch_pruning）...")
            pruner = L1Pruner(original_model, args.pruning_rate)
            pruned_model = pruner.prune(example_inputs)
            pruning_success = True
        except Exception as e:
            print(f"\n标准L1剪枝失败: {e}")
            print("\n切换到简化L1剪枝器...")
            print("注意: 简化版本将生成剪枝计划而不直接修改模型结构")
            
            pruner = SimpleL1Pruner(original_model, args.pruning_rate)
            pruned_model = pruner.prune(example_inputs)
            pruning_success = False  # 标记为未真正剪枝
    else:
        raise ValueError(f"不支持的剪枝方法: {args.pruning_method}")
    
    # 打印剪枝后模型信息
    if pruning_success:
        print_model_info(
            pruned_model,
            model_name="剪枝后模型",
            input_shape=input_shape,
            device=device
        )
        
        # 对比模型
        compare_models(
            original_model,
            pruned_model,
            input_shape=input_shape,
            device=device
        )
    else:
        print("\n注意: 由于模型包含自定义模块，未能直接修改模型结构")
        print("已生成剪枝计划，保存在统计文件中")
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.pruning_method, args.phi)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存剪枝后的模型（即使结构未改变也保存，以便后续使用）
    model_save_path = os.path.join(
        save_dir,
        f'{args.phi}_pruned_{args.pruning_rate}.pth'
    )
    pruner.save_pruned_model(model_save_path)
    
    # 保存剪枝统计信息
    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    
    # 计算FLOPs
    original_flops, _ = calculate_flops(original_model, input_shape, device)
    pruned_flops, _ = calculate_flops(pruned_model, input_shape, device)
    
    results = {
        'model_config': {
            'phi': args.phi,
            'num_classes': args.num_classes,
            'input_shape': args.input_shape,
        },
        'pruning_config': {
            'method': args.pruning_method,
            'pruning_rate': args.pruning_rate,
            'actual_pruning_applied': pruning_success,
        },
        'original_model': {
            'parameters': original_params,
            'parameters_M': round(original_params / 1e6, 2),
            'flops': original_flops,
            'flops_G': round(original_flops / 1e9, 2),
        },
        'pruned_model': {
            'parameters': pruned_params,
            'parameters_M': round(pruned_params / 1e6, 2),
            'flops': pruned_flops,
            'flops_G': round(pruned_flops / 1e9, 2),
        },
        'reduction': {
            'parameters_reduction_percent': round((1 - pruned_params / original_params) * 100, 2) if pruning_success else 0,
            'flops_reduction_percent': round((1 - pruned_flops / original_flops) * 100, 2) if original_flops > 0 and pruning_success else 0,
        },
        'model_save_path': model_save_path,
        'pruning_info': pruner.get_pruning_info(),
    }
    
    results_save_path = os.path.join(save_dir, 'pruning_results.json')
    save_pruning_results(results, results_save_path)
    
    print(f"\n{'='*60}")
    if pruning_success:
        print("剪枝完成！")
    else:
        print("剪枝分析完成！")
    print(f"{'='*60}")
    print(f"模型保存位置: {model_save_path}")
    print(f"统计结果保存位置: {results_save_path}")
    
    if not pruning_success:
        print(f"\n重要提示:")
        print(f"由于此SegFormer版本使用了自定义KAN层，torch_pruning无法直接应用。")
        print(f"已生成详细的剪枝计划，您可以:")
        print(f"1. 查看pruning_results.json中的剪枝计划")
        print(f"2. 使用该计划进行知识蒸馏训练一个更小的模型")
        print(f"3. 考虑使用标准SegFormer（不含KAN层）进行剪枝")
    else:
        print(f"\n提示: 剪枝后的模型需要进行微调以恢复精度")
        print(f"可以使用 train.py 加载剪枝后的模型进行微调:")
        print(f"  python train.py --model_path {model_save_path} --phi {args.phi} --num_classes {args.num_classes}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
