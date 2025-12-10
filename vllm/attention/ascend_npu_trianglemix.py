# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend NPU + CANN optimized TriangleMix implementation."""

import os
from typing import Optional

import torch
import torch.nn as nn

# 检测Ascend环境
IS_ASCEND_NPU = (
    'ASCEND_RT_VISIBLE_DEVICES' in os.environ or
    'ASCEND_DEVICE_ID' in os.environ or
    'ASCEND_SLOG_PRINT_TO_STDOUT' in os.environ
)

# 检测CANN可用性
try:
    import torch_npu
    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False

from vllm.attention.trianglemix_attention import (
    TriangleMixConfig,
    TriangleMixMaskGenerator,
)


class AscendNPUTriangleMixConfig(TriangleMixConfig):
    """Ascend NPU特定的TriangleMix配置。"""
    
    def __init__(
        self,
        num_sink_tokens: int = 4,
        sliding_window_size: int = 32,
        num_last_tokens: int = 64,
        num_triangle_layers: Optional[int] = None,
        triangle_layer_indices: Optional[list[int]] = None,
        # Ascend特定参数
        npu_block_size: int = 64,
        use_block_sparse: bool = True,
        use_acl_ops: bool = False,
    ):
        """
        初始化Ascend NPU TriangleMix配置。
        
        Args:
            npu_block_size: NPU块大小（推荐64或128）
            use_block_sparse: 是否使用块级稀疏（推荐长序列）
            use_acl_ops: 是否使用ACL自定义算子（需要编译）
        """
        super().__init__(
            num_sink_tokens=num_sink_tokens,
            sliding_window_size=sliding_window_size,
            num_last_tokens=num_last_tokens,
            num_triangle_layers=num_triangle_layers,
            triangle_layer_indices=triangle_layer_indices,
        )
        self.npu_block_size = npu_block_size
        self.use_block_sparse = use_block_sparse
        self.use_acl_ops = use_acl_ops


class AscendBlockSparseMaskGenerator(TriangleMixMaskGenerator):
    """Ascend NPU优化的块级稀疏掩码生成器。"""
    
    def __init__(self, config: AscendNPUTriangleMixConfig):
        """
        初始化Ascend块级稀疏掩码生成器。
        
        Args:
            config: AscendNPUTriangleMixConfig实例
        """
        super().__init__(config)
        self.npu_block_size = config.npu_block_size
        self.use_block_sparse = config.use_block_sparse
    
    def _get_block_sparse_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        为长序列生成块级稀疏掩码。
        
        用块级掩码代替token级掩码，大幅降低内存消耗：
        - Token级：O(N²) = O(4096²) = 16M个元素
        - 块级（block_size=64）：O((N/64)²) = O(64²) = 4K个元素
        
        Args:
            seq_len: 序列长度
            device: 设备
            dtype: 数据类型
            
        Returns:
            块级掩码 shape (num_blocks, num_blocks)
        """
        block_size = self.npu_block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # 在块级别转换参数
        si = max(1, self.config.num_sink_tokens // block_size)
        sl = max(1, self.config.sliding_window_size // block_size)
        last = max(1, self.config.num_last_tokens // block_size)
        
        block_mask = torch.zeros(
            (num_blocks, num_blocks),
            device=device,
            dtype=dtype,
        )
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Streaming blocks
                if i >= j and (j <= si or (i - j) <= sl):
                    block_mask[i, j] = 1
                # Last blocks
                elif i >= j and (num_blocks - i) < last and j > si:
                    block_mask[i, j] = 1
        
        return block_mask
    
    def _expand_block_mask_to_token_level(
        self,
        block_mask: torch.Tensor,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        将块级掩码扩展到token级别。
        
        仅用于与标准attention兼容，对长序列会OOM。
        
        Args:
            block_mask: 块级掩码
            seq_len: 序列长度
            device: 设备
            dtype: 数据类型
            
        Returns:
            Token级掩码
        """
        block_size = self.npu_block_size
        num_blocks = block_mask.shape[0]
        
        # 创建token级掩码
        token_mask = torch.zeros(
            (seq_len, seq_len),
            device=device,
            dtype=dtype,
        )
        
        for bi in range(num_blocks):
            for bj in range(num_blocks):
                if block_mask[bi, bj] > 0:
                    # 将块级掩码展开为token范围
                    i_start = bi * block_size
                    i_end = min((bi + 1) * block_size, seq_len)
                    j_start = bj * block_size
                    j_end = min((bj + 1) * block_size, seq_len)
                    
                    token_mask[i_start:i_end, j_start:j_end] = 1
        
        return token_mask
    
    def get_attention_mask(
        self,
        seq_len: int,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        use_triangle: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        获取Ascend NPU优化的注意力掩码。
        
        Args:
            seq_len: 序列长度
            layer_idx: 层索引
            device: 设备
            dtype: 数据类型
            use_triangle: 是否使用Triangle模式
            
        Returns:
            注意力掩码（None表示密集注意力）
        """
        # 缓存检查
        cache_key = (seq_len, layer_idx, str(device), str(dtype), use_triangle)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        # 决定掩码类型
        if use_triangle:
            if self.use_block_sparse and seq_len > 2048:
                # 长序列使用块级稀疏
                block_mask = self._get_block_sparse_mask(seq_len, device, dtype)
                # 返回块级掩码，由CANN后端处理
                attn_mask = self._convert_block_mask_for_cann(block_mask, device, dtype)
            else:
                # 短序列使用标准Triangle
                mask = self._create_triangle_mask(seq_len, device, dtype)
                attn_mask = self._mask_to_attn_mask(mask, device, dtype)
        else:
            # 密集attention
            if seq_len > 4096:
                # 极长序列，使用块级表示
                block_mask = torch.tril(
                    torch.ones(
                        ((seq_len + 63) // 64, (seq_len + 63) // 64),
                        device=device,
                        dtype=dtype,
                    )
                )
                attn_mask = self._convert_block_mask_for_cann(block_mask, device, dtype)
            else:
                mask = self._create_dense_mask(seq_len, device, dtype)
                attn_mask = self._mask_to_attn_mask(mask, device, dtype)
        
        self._mask_cache[cache_key] = attn_mask
        return attn_mask
    
    @staticmethod
    def _mask_to_attn_mask(
        mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        将掩码转换为注意力掩码格式（避免-inf以提高NPU兼容性）。
        
        Args:
            mask: 掩码张量（0/1）
            device: 设备
            dtype: 数据类型
            
        Returns:
            注意力掩码（0/-1e9）
        """
        return torch.where(
            mask > 0,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(-1e9, device=device, dtype=dtype),  # 使用-1e9而不是-inf
        )
    
    @staticmethod
    def _convert_block_mask_for_cann(
        block_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        将块级掩码转换为CANN友好的格式。
        
        Args:
            block_mask: 块级掩码
            device: 设备
            dtype: 数据类型
            
        Returns:
            CANN友好的掩码
        """
        # 转换为注意力掩码格式
        attn_mask = torch.where(
            block_mask > 0,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(-1e9, device=device, dtype=dtype),
        )
        
        # 确保在NPU设备上
        if HAS_TORCH_NPU and 'npu' in str(device):
            attn_mask = attn_mask.to('npu')
        
        return attn_mask


class AscendNPUTriangleMixAttention:
    """Ascend NPU优化的TriangleMix注意力。"""
    
    def __init__(
        self,
        config: AscendNPUTriangleMixConfig,
        num_layers: int,
    ):
        """
        初始化Ascend NPU TriangleMix注意力。
        
        Args:
            config: AscendNPUTriangleMixConfig实例
            num_layers: 层数
        """
        self.config = config
        self.num_layers = num_layers
        self.mask_generator = AscendBlockSparseMaskGenerator(config)
        
        # 自动选择Triangle层
        if config.num_triangle_layers is not None and not config.triangle_layer_indices:
            self.config.triangle_layer_indices = list(range(config.num_triangle_layers))
    
    def get_attn_mask(
        self,
        seq_len: int,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """
        获取该层的注意力掩码。
        
        Args:
            seq_len: 序列长度
            layer_idx: 层索引
            device: 设备
            dtype: 数据类型
            
        Returns:
            注意力掩码或None
        """
        use_triangle = self.mask_generator.should_use_triangle(layer_idx)
        
        # 短序列使用密集注意力
        if seq_len <= 512:
            return None
        
        return self.mask_generator.get_attention_mask(
            seq_len=seq_len,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            use_triangle=use_triangle,
        )
    
    def clear_cache(self):
        """清除掩码缓存（重要：释放NPU内存）。"""
        self.mask_generator.clear_cache()
        
        if HAS_TORCH_NPU:
            try:
                torch_npu.npu.empty_cache()
            except Exception:
                pass


def create_ascend_trianglemix_config(
    model_size: str = "14B",
    num_triangle_layers: Optional[int] = None,
    use_block_sparse: bool = True,
) -> AscendNPUTriangleMixConfig:
    """
    为Ascend NPU创建TriangleMix配置。
    
    Args:
        model_size: 模型大小（"7B", "14B", "32B"等）
        num_triangle_layers: Triangle层数
        use_block_sparse: 是否使用块级稀疏
        
    Returns:
        AscendNPUTriangleMixConfig实例
    """
    if num_triangle_layers is None:
        if model_size == "14B":
            num_triangle_layers = 12
        elif model_size == "32B":
            num_triangle_layers = 16
        else:
            num_triangle_layers = 8
    
    return AscendNPUTriangleMixConfig(
        num_sink_tokens=4,
        sliding_window_size=32,
        num_last_tokens=64,
        num_triangle_layers=num_triangle_layers,
        npu_block_size=64,  # Ascend推荐块大小
        use_block_sparse=use_block_sparse,
        use_acl_ops=False,  # 需要编译ACL时设为True
    )
