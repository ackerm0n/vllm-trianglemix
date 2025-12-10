# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TriangleMix attention mask generation for efficient long-context prefilling."""

from typing import Optional

import torch
import torch.nn as nn


class TriangleMixConfig:
    """Configuration for TriangleMix attention pattern."""
    
    def __init__(
        self,
        num_sink_tokens: int = 4,
        sliding_window_size: int = 32,
        num_last_tokens: int = 64,
        num_triangle_layers: Optional[int] = None,
        triangle_layer_indices: Optional[list[int]] = None,
    ):
        """
        Initialize TriangleMix configuration.
        
        Args:
            num_sink_tokens: Number of sink tokens (attention sink size)
            sliding_window_size: Size of the sliding window
            num_last_tokens: Number of tokens in the last section
            num_triangle_layers: Number of layers to apply Triangle attention
            triangle_layer_indices: Specific layer indices to apply Triangle attention
        """
        self.num_sink_tokens = num_sink_tokens
        self.sliding_window_size = sliding_window_size
        self.num_last_tokens = num_last_tokens
        self.num_triangle_layers = num_triangle_layers
        self.triangle_layer_indices = triangle_layer_indices or []


class TriangleMixMaskGenerator(nn.Module):
    """Generate attention masks for TriangleMix pattern."""
    
    def __init__(self, config: TriangleMixConfig):
        """
        Initialize TriangleMix mask generator.
        
        Args:
            config: TriangleMixConfig instance
        """
        super().__init__()
        self.config = config
        self._mask_cache = {}
    
    def _create_streaming_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create Streaming mask.
        
        M_streaming[i,j] = 1 if (i >= j and j <= si) or (i >= j and i - j <= sl)
                         = 0 otherwise
        
        where:
            si = num_sink_tokens
            sl = sliding_window_size
        
        Args:
            seq_len: Sequence length N
            device: Device to create tensor on
            dtype: Data type of the mask
            
        Returns:
            Streaming mask of shape (seq_len, seq_len)
        """
        si = self.config.num_sink_tokens
        sl = self.config.sliding_window_size
        
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # First condition: i >= j and j <= si (attention sink)
                if i >= j and j <= si:
                    mask[i, j] = 1
                # Second condition: i >= j and i - j <= sl (sliding window)
                elif i >= j and (i - j) <= sl:
                    mask[i, j] = 1
        
        return mask
    
    def _create_last_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create Last Q-K mask.
        
        M_last[i,j] = 1 if (i >= j and N - i < last and j > si and i - j > sl)
                    = 0 otherwise
        
        where:
            N = seq_len
            last = num_last_tokens
            si = num_sink_tokens
            sl = sliding_window_size
        
        Args:
            seq_len: Sequence length N
            device: Device to create tensor on
            dtype: Data type of the mask
            
        Returns:
            Last Q-K mask of shape (seq_len, seq_len)
        """
        N = seq_len
        last = self.config.num_last_tokens
        si = self.config.num_sink_tokens
        sl = self.config.sliding_window_size
        
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j and (N - i) < last and j > si and (i - j) > sl:
                    mask[i, j] = 1
        
        return mask
    
    def _create_middle_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create Middle Q-K mask.
        
        M_middle[i,j] = 1 if (i >= j and N - i >= last and j > si and i - j > sl)
                      = 0 otherwise
        
        where:
            N = seq_len
            last = num_last_tokens
            si = num_sink_tokens
            sl = sliding_window_size
        
        Args:
            seq_len: Sequence length N
            device: Device to create tensor on
            dtype: Data type of the mask
            
        Returns:
            Middle Q-K mask of shape (seq_len, seq_len)
        """
        N = seq_len
        last = self.config.num_last_tokens
        si = self.config.num_sink_tokens
        sl = self.config.sliding_window_size
        
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j and (N - i) >= last and j > si and (i - j) > sl:
                    mask[i, j] = 1
        
        return mask
    
    def _create_triangle_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create Triangle mask (combination of Streaming and Last, excluding Middle).
        
        Triangle pattern = Streaming + Last (without Middle Q-K section)
        
        Args:
            seq_len: Sequence length N
            device: Device to create tensor on
            dtype: Data type of the mask
            
        Returns:
            Triangle mask of shape (seq_len, seq_len)
        """
        streaming = self._create_streaming_mask(seq_len, device, dtype)
        last = self._create_last_mask(seq_len, device, dtype)
        triangle = streaming + last
        # Clamp to ensure values are 0 or 1
        triangle = torch.clamp(triangle, 0, 1)
        
        return triangle
    
    def _create_dense_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create dense (causal) mask - full attention.
        
        Args:
            seq_len: Sequence length N
            device: Device to create tensor on
            dtype: Data type of the mask
            
        Returns:
            Dense causal mask of shape (seq_len, seq_len)
        """
        # Causal mask: lower triangular matrix
        return torch.tril(torch.ones((seq_len, seq_len), 
                                     device=device, dtype=dtype))
    
    def get_attention_mask(
        self,
        seq_len: int,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        use_triangle: bool = False,
    ) -> torch.Tensor:
        """
        Get attention mask for a specific layer.
        
        Args:
            seq_len: Sequence length
            layer_idx: Current layer index
            device: Device to create tensor on
            dtype: Data type of the mask
            use_triangle: Whether to use Triangle pattern for this layer
            
        Returns:
            Attention mask of shape (seq_len, seq_len)
        """
        cache_key = (seq_len, layer_idx, str(device), str(dtype), use_triangle)
        
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        if use_triangle:
            mask = self._create_triangle_mask(seq_len, device, dtype)
        else:
            mask = self._create_dense_mask(seq_len, device, dtype)
        
        # Convert to attention mask format (0 for attended, -inf for masked)
        # In vLLM, we typically use additive masks
        attn_mask = torch.where(
            mask > 0,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float('-inf'), device=device, dtype=dtype)
        )
        
        self._mask_cache[cache_key] = attn_mask
        return attn_mask
    
    def should_use_triangle(self, layer_idx: int) -> bool:
        """
        Check if Triangle attention should be used for the given layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            True if Triangle attention should be used, False otherwise
        """
        if self.config.triangle_layer_indices:
            return layer_idx in self.config.triangle_layer_indices
        elif self.config.num_triangle_layers is not None:
            # Apply Triangle to the first num_triangle_layers
            return layer_idx < self.config.num_triangle_layers
        return False
    
    def clear_cache(self):
        """Clear the mask cache."""
        self._mask_cache.clear()


class TriangleMixAttention:
    """Helper class to apply TriangleMix to attention mechanism."""
    
    def __init__(
        self,
        config: TriangleMixConfig,
        num_layers: int,
    ):
        """
        Initialize TriangleMix attention helper.
        
        Args:
            config: TriangleMixConfig instance
            num_layers: Total number of layers in the model
        """
        self.config = config
        self.num_layers = num_layers
        self.mask_generator = TriangleMixMaskGenerator(config)
        
        # Auto-select Triangle layers based on contribution analysis
        if config.num_triangle_layers is not None and not config.triangle_layer_indices:
            # Default strategy: apply Triangle to the first num_triangle_layers
            self.config.triangle_layer_indices = list(range(config.num_triangle_layers))
    
    def get_attn_mask(
        self,
        seq_len: int,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """
        Get attention mask for the given layer.
        
        Args:
            seq_len: Sequence length
            layer_idx: Current layer index
            device: Device to create tensor on
            dtype: Data type of the mask
            
        Returns:
            Attention mask if needed, None for dense attention
        """
        use_triangle = self.mask_generator.should_use_triangle(layer_idx)
        
        if seq_len <= 2048:
            # For short sequences, use dense attention
            return None
        
        return self.mask_generator.get_attention_mask(
            seq_len=seq_len,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            use_triangle=use_triangle,
        )


# Gradient-based analysis for selecting Triangle layers
class TriangleMixAnalyzer(nn.Module):
    """Analyze Middle Q-K contributions for selecting Triangle layers."""
    
    def __init__(self, num_layers: int):
        """
        Initialize TriangleMix analyzer.
        
        Args:
            num_layers: Number of layers in the model
        """
        super().__init__()
        self.num_layers = num_layers
        self.middle_qk_gradients = [[] for _ in range(num_layers)]
    
    def record_middle_qk_gradient(
        self,
        layer_idx: int,
        gradient: torch.Tensor,
    ):
        """
        Record Middle Q-K gradient for a layer.
        
        Args:
            layer_idx: Layer index
            gradient: Gradient tensor for Middle Q-K section
        """
        if layer_idx < self.num_layers:
            self.middle_qk_gradients[layer_idx].append(
                gradient.detach().cpu().mean().item()
            )
    
    def get_triangle_layers(
        self,
        num_triangle_layers: int,
    ) -> list[int]:
        """
        Get layer indices to apply Triangle attention based on gradient analysis.
        
        Selects the layers with the lowest Middle Q-K contribution gradients.
        
        Args:
            num_triangle_layers: Number of layers to convert to Triangle attention
            
        Returns:
            List of layer indices for Triangle attention
        """
        # Calculate average gradient for each layer
        avg_gradients = []
        for layer_idx in range(self.num_layers):
            if self.middle_qk_gradients[layer_idx]:
                avg_grad = sum(self.middle_qk_gradients[layer_idx]) / len(
                    self.middle_qk_gradients[layer_idx]
                )
            else:
                avg_grad = float('inf')  # High value if no gradients recorded
            avg_gradients.append((layer_idx, avg_grad))
        
        # Sort by gradient (ascending) and select the lowest num_triangle_layers
        sorted_layers = sorted(avg_gradients, key=lambda x: x[1])
        triangle_layers = [idx for idx, _ in sorted_layers[:num_triangle_layers]]
        
        return sorted(triangle_layers)
    
    def reset(self):
        """Reset recorded gradients."""
        self.middle_qk_gradients = [[] for _ in range(self.num_layers)]
