#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU + CANN TriangleMix使用示例。"""

import os
import torch

# 确保检测到Ascend环境
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'  # 使用NPU 0

from vllm.attention.ascend_npu_trianglemix import (
    AscendNPUTriangleMixConfig,
    AscendNPUTriangleMixAttention,
    create_ascend_trianglemix_config,
    IS_ASCEND_NPU,
    HAS_TORCH_NPU,
)


def example_1_basic_usage():
    """示例1：基础使用。"""
    print("=" * 60)
    print("示例1：Ascend NPU TriangleMix基础使用")
    print("=" * 60)
    
    print(f"检测到Ascend NPU: {IS_ASCEND_NPU}")
    print(f"torch_npu可用: {HAS_TORCH_NPU}")
    
    # 创建配置
    config = create_ascend_trianglemix_config(
        model_size="14B",
        num_triangle_layers=12,
        use_block_sparse=True,
    )
    
    print(f"\n配置参数:")
    print(f"  - Sink tokens: {config.num_sink_tokens}")
    print(f"  - Sliding window: {config.sliding_window_size}")
    print(f"  - Last tokens: {config.num_last_tokens}")
    print(f"  - Triangle layers: {config.num_triangle_layers}")
    print(f"  - NPU block size: {config.npu_block_size}")
    print(f"  - Use block sparse: {config.use_block_sparse}")
    
    return config


def example_2_mask_generation():
    """示例2：掩码生成。"""
    print("\n" + "=" * 60)
    print("示例2：Ascend NPU掩码生成")
    print("=" * 60)
    
    config = create_ascend_trianglemix_config("14B")
    attn = AscendNPUTriangleMixAttention(config, num_layers=32)
    
    # 测试不同的序列长度
    seq_lengths = [512, 2048, 4096]
    device = torch.device('npu:0' if HAS_TORCH_NPU else 'cpu')
    
    for seq_len in seq_lengths:
        mask = attn.get_attn_mask(
            seq_len=seq_len,
            layer_idx=0,  # Triangle层
            device=device,
        )
        
        if mask is not None:
            attended = (mask == 0).sum().item()
            total = mask.numel()
            print(f"seq_len={seq_len:4d}: shape={mask.shape}, "
                  f"attended={attended}/{total} ({attended/total*100:.1f}%)")
        else:
            print(f"seq_len={seq_len:4d}: 使用密集注意力（无掩码）")


def example_3_layer_configuration():
    """示例3：层级配置。"""
    print("\n" + "=" * 60)
    print("示例3：Ascend NPU层级配置")
    print("=" * 60)
    
    config = create_ascend_trianglemix_config(
        model_size="14B",
        num_triangle_layers=12,
    )
    
    attn = AscendNPUTriangleMixAttention(config, num_layers=32)
    
    print("层级注意力配置（14B Qwen3）:")
    for layer_idx in range(32):
        use_triangle = attn.mask_generator.should_use_triangle(layer_idx)
        pattern = "Triangle" if use_triangle else "Dense"
        marker = "  ✓" if use_triangle else "  "
        print(f"{marker} Layer {layer_idx:2d}: {pattern}")


def example_4_memory_efficiency():
    """示例4：内存效率对比。"""
    print("\n" + "=" * 60)
    print("示例4：内存效率对比（块级 vs Token级）")
    print("=" * 60)
    
    seq_len = 4096
    block_size = 64
    
    # Token级掩码内存
    token_level_elements = seq_len * seq_len
    token_level_memory_mb = token_level_elements * 4 / (1024 * 1024)  # 32-bit float
    
    # 块级掩码内存
    num_blocks = (seq_len + block_size - 1) // block_size
    block_level_elements = num_blocks * num_blocks
    block_level_memory_mb = block_level_elements * 4 / (1024 * 1024)
    
    compression = token_level_memory_mb / block_level_memory_mb
    
    print(f"序列长度: {seq_len}")
    print(f"块大小: {block_size}")
    print()
    print(f"Token级掩码:")
    print(f"  - 元素数: {token_level_elements:,}")
    print(f"  - 内存使用: {token_level_memory_mb:.2f} MB")
    print()
    print(f"块级掩码:")
    print(f"  - 元素数: {block_level_elements:,}")
    print(f"  - 内存使用: {block_level_memory_mb:.2f} MB")
    print()
    print(f"压缩比: {compression:.1f}x")
    print(f"内存节省: {token_level_memory_mb - block_level_memory_mb:.2f} MB")


def example_5_performance_implications():
    """示例5：性能影响分析。"""
    print("\n" + "=" * 60)
    print("示例5：性能影响分析")
    print("=" * 60)
    
    config = create_ascend_trianglemix_config("14B", num_triangle_layers=12)
    
    print("预期性能改善（相对于密集注意力）:")
    print()
    print("短序列（<1024 tokens）:")
    print("  - Triangle overhead：~2-5%（掩码生成开销）")
    print("  - 建议：使用密集注意力")
    print()
    print("中长序列（1024-4096 tokens）:")
    print("  - FLOP削减：~30-40%（跳过Middle Q-K）")
    print("  - 预期加速：~1.5-2x")
    print("  - 建议：使用Triangle模式")
    print()
    print("极长序列（>4096 tokens）:")
    print("  - FLOP削减：~40-50%")
    print("  - 内存削减：~50-60%（块级掩码）")
    print("  - 预期加速：~2-3x")
    print("  - 建议：使用块级稀疏Triangle模式")
    print()
    print(f"TriangleMix配置:")
    print(f"  - Sink tokens: {config.num_sink_tokens} (保持关键历史)")
    print(f"  - Sliding window: {config.sliding_window_size} (局部上下文)")
    print(f"  - Last tokens: {config.num_last_tokens} (最后关键信息)")
    print(f"  - Triangle layers: 12/32 (~38%的层使用稀疏模式)")


def example_6_integration_with_qwen3():
    """示例6：与Qwen3集成。"""
    print("\n" + "=" * 60)
    print("示例6：与Qwen3模型集成")
    print("=" * 60)
    
    print("在Qwen3模型中使用Ascend NPU TriangleMix的步骤:")
    print()
    print("步骤1：导入所需模块")
    print("```python")
    print("from vllm.attention.ascend_npu_trianglemix import (")
    print("    create_ascend_trianglemix_config,")
    print("    AscendNPUTriangleMixAttention,")
    print(")")
    print("```")
    print()
    print("步骤2：创建配置")
    print("```python")
    print("trianglemix_config = create_ascend_trianglemix_config(")
    print('    model_size="14B",')
    print("    num_triangle_layers=12,")
    print("    use_block_sparse=True,")
    print(")")
    print("```")
    print()
    print("步骤3：在Qwen3Attention中集成")
    print("```python")
    print("self.trianglemix_attn = AscendNPUTriangleMixAttention(")
    print("    config=trianglemix_config,")
    print("    num_layers=num_layers,")
    print(")")
    print("```")
    print()
    print("步骤4：在forward中使用")
    print("```python")
    print("if self.trianglemix_attn is not None:")
    print("    attn_mask = self.trianglemix_attn.get_attn_mask(")
    print("        seq_len=q.shape[1],")
    print("        layer_idx=self.layer_idx,")
    print("        device=q.device,")
    print("    )")
    print("    # 将attn_mask传递给attention计算")
    print("```")


def example_7_troubleshooting():
    """示例7：故障排除。"""
    print("\n" + "=" * 60)
    print("示例7：常见问题与解决方案")
    print("=" * 60)
    
    print("问题1：未检测到Ascend NPU")
    print("  原因：环境变量未设置")
    print("  解决：export ASCEND_RT_VISIBLE_DEVICES=0")
    print()
    
    print("问题2：torch_npu不可用")
    print("  原因：未安装torch_npu")
    print("  解决：pip install torch_npu")
    print()
    
    print("问题3：OOM（超出内存）")
    print("  原因：序列过长导致掩码矩阵过大")
    print("  解决：")
    print("    - 启用block_sparse: use_block_sparse=True")
    print("    - 减少num_last_tokens")
    print("    - 增加num_triangle_layers")
    print()
    
    print("问题4：精度下降")
    print("  原因：不合适的层选择")
    print("  解决：")
    print("    - 减少num_triangle_layers")
    print("    - 调整num_last_tokens")
    print()
    
    print("问题5：性能没有改善")
    print("  原因：序列过短或配置不合适")
    print("  解决：")
    print("    - 用长序列测试（>2048 tokens）")
    print("    - 检查层选择是否正确")
    print("    - 验证NPU使用率（npu-smi命令）")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ascend NPU + CANN TriangleMix 完整示例")
    print("=" * 60)
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_mask_generation()
        example_3_layer_configuration()
        example_4_memory_efficiency()
        example_5_performance_implications()
        example_6_integration_with_qwen3()
        example_7_troubleshooting()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
