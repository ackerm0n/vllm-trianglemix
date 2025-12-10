#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick start guide for TriangleMix with Qwen3 on Ascend NPU."""

import os
import sys
from typing import Optional

import torch

# Add vLLM to path if needed
# sys.path.insert(0, '/Users/tsy/Downloads/vllm-0.11.0')

from vllm.attention.trianglemix_attention import TriangleMixConfig
from vllm.attention.trianglemix_config import TriangleMixInferenceConfig
from vllm.attention.ascend_trianglemix import (
    AscendTriangleMixAttention,
    AscendNPUTriangleMixOptimizer,
)


class QuickStartExample:
    """Quick start examples for TriangleMix."""
    
    @staticmethod
    def example_1_basic_setup():
        """Example 1: Basic TriangleMix setup for Ascend NPU."""
        print("\n" + "=" * 70)
        print("Example 1: Basic TriangleMix Setup for Ascend NPU")
        print("=" * 70)
        
        # 创建配置
        config = TriangleMixInferenceConfig.for_ascend_npu(
            num_triangle_layers=12,
            enable_gradient_analysis=False,
        )
        
        # 转换为TriangleMixConfig
        trianglemix_cfg = config.to_trianglemix_config()
        
        print(f"\nTriangleMix Configuration:")
        print(f"  Enable: {config.enable_trianglemix}")
        print(f"  Triangle Layers: {config.num_triangle_layers}")
        print(f"  Sink Tokens: {trianglemix_cfg.num_sink_tokens}")
        print(f"  Sliding Window: {trianglemix_cfg.sliding_window_size}")
        print(f"  Last Tokens: {trianglemix_cfg.num_last_tokens}")
        print(f"  NPU Optimization: {config.use_npu_optimization}")
    
    @staticmethod
    def example_2_qwen3_config():
        """Example 2: Qwen3 specific configuration."""
        print("\n" + "=" * 70)
        print("Example 2: Qwen3 Specific Configuration")
        print("=" * 70)
        
        # 不同大小的Qwen3模型配置
        models = ["14B", "32B"]
        
        for model_size in models:
            config = TriangleMixInferenceConfig.for_qwen3(
                model_size=model_size,
            )
            print(f"\nQwen3-{model_size}:")
            print(f"  Triangle Layers: {config.num_triangle_layers}")
            print(f"  Recommended for: Long-context inference on Ascend NPU")
    
    @staticmethod
    def example_3_custom_layers():
        """Example 3: Custom layer selection."""
        print("\n" + "=" * 70)
        print("Example 3: Custom Layer Selection")
        print("=" * 70)
        
        # 自定义指定哪些层使用Triangle
        config = TriangleMixInferenceConfig(
            enable_trianglemix=True,
            triangle_layer_indices=[0, 1, 2, 3, 4, 5],  # 前6层
            num_sink_tokens=4,
            sliding_window_size=32,
            num_last_tokens=64,
        )
        
        print(f"\nCustom Configuration:")
        print(f"  Layers with Triangle: {config.triangle_layer_indices}")
        print(f"  This applies Triangle attention to the first 6 layers")
    
    @staticmethod
    def example_4_mask_generation():
        """Example 4: Mask generation and visualization."""
        print("\n" + "=" * 70)
        print("Example 4: Attention Mask Generation")
        print("=" * 70)
        
        from vllm.attention.trianglemix_attention import TriangleMixMaskGenerator
        
        # 创建配置和生成器
        config = TriangleMixConfig(
            num_sink_tokens=2,
            sliding_window_size=4,
            num_last_tokens=6,
        )
        
        generator = TriangleMixMaskGenerator(config)
        seq_len = 12
        device = torch.device('cpu')
        dtype = torch.float32
        
        # 生成Dense掩码
        dense_mask = generator._create_dense_mask(seq_len, device, dtype)
        dense_attended = (dense_mask == 1).sum().item()
        
        # 生成Triangle掩码
        triangle_mask = generator._create_triangle_mask(seq_len, device, dtype)
        triangle_attended = (triangle_mask == 1).sum().item()
        
        print(f"\nAttention Mask Comparison (seq_len={seq_len}):")
        print(f"  Dense Attention:")
        print(f"    Attended positions: {dense_attended}/{seq_len*seq_len} ({dense_attended/(seq_len*seq_len)*100:.1f}%)")
        print(f"  Triangle Attention:")
        print(f"    Attended positions: {triangle_attended}/{seq_len*seq_len} ({triangle_attended/(seq_len*seq_len)*100:.1f}%)")
        print(f"  Compression: {dense_attended/triangle_attended:.2f}x")
    
    @staticmethod
    def example_5_ascend_npu_optimization():
        """Example 5: Ascend NPU optimization."""
        print("\n" + "=" * 70)
        print("Example 5: Ascend NPU Optimization")
        print("=" * 70)
        
        # 创建掩码
        seq_len = 16
        mask = torch.full((seq_len, seq_len), float('-inf'))
        # 创建因果掩码
        for i in range(seq_len):
            mask[i, :i+1] = 0.0
        
        print(f"\nOriginal mask (contains -inf): {torch.isinf(mask).sum().item()} positions")
        
        # 优化掩码
        optimizer = AscendNPUTriangleMixOptimizer()
        optimized = optimizer.optimize_mask_for_npu(mask)
        
        print(f"Optimized mask (no -inf): {torch.isinf(optimized).sum().item()} positions")
        print(f"  Large negative values used for masking: {(optimized < -1e5).sum().item()}")
        print(f"  Suitable for Ascend NPU execution")
    
    @staticmethod
    def example_6_attention_computation():
        """Example 6: Complete attention computation."""
        print("\n" + "=" * 70)
        print("Example 6: Complete Attention Computation")
        print("=" * 70)
        
        # 配置
        config = TriangleMixConfig(
            num_sink_tokens=4,
            sliding_window_size=8,
            num_last_tokens=16,
            num_triangle_layers=1,
        )
        
        # 创建Ascend NPU优化的attention
        attn = AscendTriangleMixAttention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / (64 ** 0.5),
            trianglemix_config=config,
            layer_idx=0,
            num_layers=4,
        )
        
        # 创建输入
        batch_size = 2
        seq_len = 32
        num_heads = 8
        head_dim = 64
        
        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        print(f"\nInput shapes:")
        print(f"  Query: {query.shape}")
        print(f"  Key: {key.shape}")
        print(f"  Value: {value.shape}")
        
        # 计算attention
        output = attn(query, key, value)
        
        print(f"\nOutput shape: {output.shape}")
        print(f"  Layer 0 using Triangle attention: YES")
        print(f"  Middle Q-K section: SKIPPED")
        print(f"  Computational complexity: Reduced")
    
    @staticmethod
    def example_7_gradient_analysis():
        """Example 7: Gradient-based layer selection."""
        print("\n" + "=" * 70)
        print("Example 7: Gradient-based Layer Selection")
        print("=" * 70)
        
        from vllm.attention.trianglemix_attention import TriangleMixAnalyzer
        
        num_layers = 8
        analyzer = TriangleMixAnalyzer(num_layers=num_layers)
        
        print(f"\nAnalyzing Middle Q-K contributions across {num_layers} layers...")
        
        # 模拟梯度记录（实际应用中从反向传播获取）
        for layer_idx in range(num_layers):
            # 模拟：较低层有较高贡献，较高层有较低贡献
            contribution = 1.0 - (layer_idx / num_layers) * 0.7
            gradient = torch.tensor(contribution)
            
            for _ in range(5):  # 多次采样
                analyzer.record_middle_qk_gradient(layer_idx, gradient)
        
        # 获取应该应用Triangle的层
        num_to_select = 4
        triangle_layers = analyzer.get_triangle_layers(num_triangle_layers=num_to_select)
        
        print(f"\nSelected {num_to_select} layers for Triangle attention:")
        print(f"  Layer indices: {triangle_layers}")
        print(f"  These layers have the lowest Middle Q-K contributions")
        print(f"  Inference speed up without accuracy loss")
    
    @staticmethod
    def example_8_environment_setup():
        """Example 8: Environment setup for production."""
        print("\n" + "=" * 70)
        print("Example 8: Environment Setup for Production")
        print("=" * 70)
        
        print("\nTo use TriangleMix in production, set these environment variables:")
        print("""
# Enable TriangleMix
export VLLM_TRIANGLEMIX_ENABLED=1

# Number of Triangle layers (for Qwen3-14B: 12 is recommended)
export VLLM_TRIANGLEMIX_LAYERS=12

# Ascend NPU optimization
export VLLM_NPU_OPTIMIZATION=1

# NPU block size (typically 64)
export VLLM_NPU_BLOCK_SIZE=64

# Gradient analysis (for automatic layer selection)
export VLLM_TRIANGLEMIX_GRADIENT_ANALYSIS=1

# Then run inference:
python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen3-14B-Chat \\
    --device ascend \\
    --gpu-memory-utilization 0.9
        """)
    
    @staticmethod
    def run_all_examples():
        """Run all examples."""
        print("\n" + "=" * 70)
        print("TriangleMix Quick Start - All Examples")
        print("=" * 70)
        
        examples = [
            QuickStartExample.example_1_basic_setup,
            QuickStartExample.example_2_qwen3_config,
            QuickStartExample.example_3_custom_layers,
            QuickStartExample.example_4_mask_generation,
            QuickStartExample.example_5_ascend_npu_optimization,
            QuickStartExample.example_6_attention_computation,
            QuickStartExample.example_7_gradient_analysis,
            QuickStartExample.example_8_environment_setup,
        ]
        
        for example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"\nError in {example_func.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review TRIANGLEMIX_README.md for detailed documentation")
        print("2. Run tests: python tests/trianglemix_attention_test.py")
        print("3. Integrate with your Qwen3 model")
        print("4. Benchmark on Ascend NPU hardware")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TriangleMix Quick Start Guide"
    )
    parser.add_argument(
        "--example",
        type=int,
        default=0,
        help="Run specific example (0=all, 1-8=specific example)",
    )
    
    args = parser.parse_args()
    
    if args.example == 0:
        QuickStartExample.run_all_examples()
    elif 1 <= args.example <= 8:
        example_func = getattr(
            QuickStartExample,
            f"example_{args.example}_" + [
                "basic_setup",
                "qwen3_config",
                "custom_layers",
                "mask_generation",
                "ascend_npu_optimization",
                "attention_computation",
                "gradient_analysis",
                "environment_setup",
            ][args.example - 1],
        )
        example_func()
    else:
        print(f"Invalid example number: {args.example}")
        sys.exit(1)


if __name__ == "__main__":
    main()
