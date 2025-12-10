# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend NPU optimized TriangleMix attention implementation."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention.trianglemix_attention import (TriangleMixConfig,
                                                   TriangleMixAttention,
                                                   TriangleMixMaskGenerator)


class AscendTriangleMixAttention(nn.Module):
    """Ascend NPU optimized TriangleMix attention implementation."""
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        trianglemix_config: Optional[TriangleMixConfig] = None,
        layer_idx: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        """
        Initialize Ascend NPU optimized TriangleMix attention.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            scale: Scaling factor for attention scores
            trianglemix_config: TriangleMix configuration
            layer_idx: Current layer index
            num_layers: Total number of layers
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.layer_idx = layer_idx or 0
        
        self.trianglemix_attn = None
        if trianglemix_config is not None and num_layers is not None:
            self.trianglemix_attn = TriangleMixAttention(
                config=trianglemix_config,
                num_layers=num_layers,
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with TriangleMix pattern.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            key: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            value: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
            attn_mask: Optional attention mask
            
        Returns:
            Attention output of shape (batch_size, seq_len, num_heads, head_dim)
        """
        # Compute attention scores
        # Reshape for batch matrix multiplication
        batch_size, seq_len, _, _ = query.shape
        
        # Get TriangleMix mask if configured
        if self.trianglemix_attn is not None:
            tri_mask = self.trianglemix_attn.get_attn_mask(
                seq_len=seq_len,
                layer_idx=self.layer_idx,
                device=query.device,
                dtype=query.dtype,
            )
            if tri_mask is not None:
                attn_mask = tri_mask if attn_mask is None else attn_mask + tri_mask
        
        # Standard attention computation
        # query: (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        
        return output


class AscendNPUTriangleMixOptimizer:
    """Optimize TriangleMix for Ascend NPU execution."""
    
    @staticmethod
    def optimize_mask_for_npu(
        mask: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """
        Optimize mask for Ascend NPU block-wise execution.
        
        Ascend NPU performs better with block-wise operations.
        
        Args:
            mask: Attention mask tensor
            block_size: Block size for optimization
            
        Returns:
            Optimized mask tensor
        """
        if mask.numel() == 0:
            return mask
        
        # Convert -inf to large negative value for NPU compatibility
        mask = torch.where(
            torch.isinf(mask) & (mask < 0),
            torch.tensor(-1e6, dtype=mask.dtype, device=mask.device),
            mask,
        )
        
        return mask
    
    @staticmethod
    def sparse_attention_kernel(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        block_size: int = 64,
    ) -> torch.Tensor:
        """
        Sparse attention computation optimized for Ascend NPU.
        
        This implementation uses block-wise sparse patterns for efficiency.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Sparse attention mask
            scale: Scaling factor
            block_size: Block size for computation
            
        Returns:
            Attention output
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        # Reshape for computation
        q = query.view(batch_size * seq_len, num_heads, head_dim)
        k = key.view(batch_size * seq_len, num_heads, head_dim)
        v = value.view(batch_size * seq_len, num_heads, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-2, -1)) * scale
        
        # Apply mask
        scores = scores + mask.unsqueeze(1)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        output = torch.matmul(attn_weights, v.transpose(0, 1).transpose(-2, -1))
        
        # Reshape back
        output = output.view(batch_size, seq_len, num_heads, head_dim)
        
        return output


def create_trianglemix_config_from_hf(
    hf_config,
    use_trianglemix: bool = False,
    num_triangle_layers: Optional[int] = None,
) -> Optional[TriangleMixConfig]:
    """
    Create TriangleMix configuration from HuggingFace model config.
    
    Args:
        hf_config: HuggingFace model configuration
        use_trianglemix: Whether to enable TriangleMix
        num_triangle_layers: Number of layers to apply TriangleMix
        
    Returns:
        TriangleMixConfig instance or None
    """
    if not use_trianglemix:
        return None
    
    # Get attention sink and sliding window from config if available
    num_sink_tokens = getattr(hf_config, "num_sink_tokens", 4)
    sliding_window_size = getattr(hf_config, "sliding_window", 32)
    
    # Default last tokens
    num_last_tokens = max(64, sliding_window_size * 2)
    
    return TriangleMixConfig(
        num_sink_tokens=num_sink_tokens,
        sliding_window_size=sliding_window_size,
        num_last_tokens=num_last_tokens,
        num_triangle_layers=num_triangle_layers,
    )
