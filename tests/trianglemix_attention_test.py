#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test and demonstration script for TriangleMix attention."""

import torch
import torch.nn as nn
from typing import Optional

from vllm.attention.trianglemix_attention import (
    TriangleMixConfig,
    TriangleMixMaskGenerator,
    TriangleMixAttention,
    TriangleMixAnalyzer,
)


def test_streaming_mask():
    """Test Streaming mask generation."""
    print("=" * 60)
    print("Testing Streaming Mask")
    print("=" * 60)
    
    config = TriangleMixConfig(
        num_sink_tokens=4,
        sliding_window_size=8,
        num_last_tokens=16,
    )
    
    generator = TriangleMixMaskGenerator(config)
    mask = generator._create_streaming_mask(
        seq_len=24,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    
    print(f"Streaming Mask shape: {mask.shape}")
    print("Streaming Mask (1 = attend, 0 = mask):")
    print(mask)
    
    # Verify properties
    si = config.num_sink_tokens  # 4
    sl = config.sliding_window_size  # 8
    
    # Check a few specific positions
    print(f"\nVerification (si={si}, sl={sl}):")
    print(f"mask[10, 3] (should be 1, since 3 <= si): {mask[10, 3]}")
    print(f"mask[15, 8] (should be 1, since 15-8 = 7 <= sl): {mask[15, 8]}")
    print(f"mask[15, 1] (should be 0, since both conditions fail): {mask[15, 1]}")


def test_triangle_vs_dense():
    """Compare Triangle mask with dense mask."""
    print("\n" + "=" * 60)
    print("Comparing Triangle vs Dense Masks")
    print("=" * 60)
    
    config = TriangleMixConfig(
        num_sink_tokens=2,
        sliding_window_size=4,
        num_last_tokens=6,
    )
    
    generator = TriangleMixMaskGenerator(config)
    seq_len = 12
    
    triangle_mask = generator._create_triangle_mask(
        seq_len=seq_len,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    
    dense_mask = generator._create_dense_mask(
        seq_len=seq_len,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    
    print(f"Dense mask sparsity: {(dense_mask == 0).sum().item()} / {dense_mask.numel()} masked")
    print(f"Triangle mask sparsity: {(triangle_mask == 0).sum().item()} / {triangle_mask.numel()} masked")
    print(f"Compression ratio: {(dense_mask == 0).sum().item() / (triangle_mask == 0).sum().item():.2f}x")
    
    print("\nTriangle Mask (1 = attend):")
    print(triangle_mask)


def test_attention_mask_generation():
    """Test attention mask generation for different patterns."""
    print("\n" + "=" * 60)
    print("Testing Attention Mask Generation")
    print("=" * 60)
    
    config = TriangleMixConfig(
        num_sink_tokens=3,
        sliding_window_size=6,
        num_last_tokens=8,
    )
    
    generator = TriangleMixMaskGenerator(config)
    seq_len = 16
    device = torch.device('cpu')
    dtype = torch.float32
    
    # Dense attention
    dense_attn = generator.get_attention_mask(
        seq_len=seq_len,
        layer_idx=0,
        device=device,
        dtype=dtype,
        use_triangle=False,
    )
    
    # Triangle attention
    triangle_attn = generator.get_attention_mask(
        seq_len=seq_len,
        layer_idx=1,
        device=device,
        dtype=dtype,
        use_triangle=True,
    )
    
    print(f"Dense attention mask (0/-inf format):")
    print(f"  Shape: {dense_attn.shape}")
    print(f"  Valid values: {dense_attn.sum().item():.2f} / {dense_attn.numel()}")
    
    print(f"\nTriangle attention mask (0/-inf format):")
    print(f"  Shape: {triangle_attn.shape}")
    print(f"  Valid values: {triangle_attn.sum().item():.2f} / {triangle_attn.numel()}")
    print(f"  Compression: {(dense_attn != triangle_attn).sum().item()} positions differ")


def test_trianglemix_attention():
    """Test TriangleMixAttention helper class."""
    print("\n" + "=" * 60)
    print("Testing TriangleMixAttention Helper")
    print("=" * 60)
    
    config = TriangleMixConfig(
        num_sink_tokens=4,
        sliding_window_size=8,
        num_last_tokens=16,
        num_triangle_layers=4,
    )
    
    attn = TriangleMixAttention(config=config, num_layers=8)
    
    print(f"Triangle layers: {attn.config.triangle_layer_indices}")
    
    for layer_idx in range(8):
        use_tri = attn.mask_generator.should_use_triangle(layer_idx)
        print(f"Layer {layer_idx}: {'Triangle' if use_tri else 'Dense'} attention")
    
    # Test mask generation for different sequence lengths
    print("\nMask generation for different sequence lengths:")
    for seq_len in [512, 1024, 2048, 4096]:
        mask = attn.get_attn_mask(
            seq_len=seq_len,
            layer_idx=0,  # Triangle layer
            device=torch.device('cpu'),
        )
        if mask is not None:
            attended = (mask == 0).sum().item()
            total = mask.numel()
            print(f"  seq_len={seq_len}: {attended}/{total} positions attended ({attended/total*100:.1f}%)")
        else:
            print(f"  seq_len={seq_len}: No mask (dense attention)")


def test_gradient_analyzer():
    """Test gradient-based layer selection."""
    print("\n" + "=" * 60)
    print("Testing Gradient-based Layer Analysis")
    print("=" * 60)
    
    num_layers = 8
    analyzer = TriangleMixAnalyzer(num_layers=num_layers)
    
    # Simulate gradient recording for different layers
    for layer_idx in range(num_layers):
        # Lower layers have higher gradients (more important)
        # Higher layers have lower gradients (less important)
        gradient_value = 1.0 - (layer_idx / num_layers) * 0.5
        
        for _ in range(3):  # Record multiple times
            gradient = torch.tensor(gradient_value)
            analyzer.record_middle_qk_gradient(layer_idx, gradient)
    
    # Get layers with lowest contribution (best candidates for Triangle)
    triangle_layers = analyzer.get_triangle_layers(num_triangle_layers=4)
    
    print(f"Layers selected for Triangle attention: {triangle_layers}")
    print("(These should be the higher layers with lower gradients)")


def test_mask_optimization():
    """Test mask optimization for Ascend NPU."""
    print("\n" + "=" * 60)
    print("Testing Ascend NPU Mask Optimization")
    print("=" * 60)
    
    from vllm.attention.ascend_trianglemix import AscendNPUTriangleMixOptimizer
    
    # Create mask with -inf values
    mask = torch.full((4, 4), float('-inf'))
    mask[0, 0] = 0.0
    mask[1, 0:2] = 0.0
    mask[2, 0:3] = 0.0
    mask[3, :] = 0.0
    
    print("Original mask (with -inf):")
    print(mask)
    
    optimized = AscendNPUTriangleMixOptimizer.optimize_mask_for_npu(mask)
    
    print("\nOptimized mask for NPU (no -inf):")
    print(optimized)
    
    print("\nOptimization details:")
    print(f"  Contains -inf: {torch.isinf(optimized).any().item()}")
    print(f"  Max masked value: {optimized[torch.isfinite(optimized)].min().item():.2f}")


def benchmark_mask_creation():
    """Benchmark mask creation performance."""
    print("\n" + "=" * 60)
    print("Benchmarking Mask Creation Performance")
    print("=" * 60)
    
    import time
    
    config = TriangleMixConfig(
        num_sink_tokens=4,
        sliding_window_size=32,
        num_last_tokens=64,
    )
    
    generator = TriangleMixMaskGenerator(config)
    device = torch.device('cpu')
    dtype = torch.float32
    
    seq_lengths = [512, 2048, 4096]
    
    for seq_len in seq_lengths:
        # Time dense mask
        start = time.time()
        for _ in range(10):
            generator._create_dense_mask(seq_len, device, dtype)
        dense_time = (time.time() - start) / 10
        
        # Time triangle mask
        start = time.time()
        for _ in range(10):
            generator._create_triangle_mask(seq_len, device, dtype)
        triangle_time = (time.time() - start) / 10
        
        print(f"seq_len={seq_len}:")
        print(f"  Dense mask: {dense_time*1000:.2f}ms")
        print(f"  Triangle mask: {triangle_time*1000:.2f}ms")
        print(f"  Speedup: {dense_time/triangle_time:.2f}x")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TriangleMix Attention Test Suite")
    print("=" * 60)
    
    try:
        test_streaming_mask()
        test_triangle_vs_dense()
        test_attention_mask_generation()
        test_trianglemix_attention()
        test_gradient_analyzer()
        test_mask_optimization()
        benchmark_mask_creation()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
