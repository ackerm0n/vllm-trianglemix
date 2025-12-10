# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration and examples for TriangleMix attention in vLLM."""

from dataclasses import dataclass, field
from typing import Optional, List

from vllm.attention.trianglemix_attention import TriangleMixConfig


@dataclass
class TriangleMixInferenceConfig:
    """Configuration for TriangleMix inference on Ascend NPU."""
    
    # TriangleMix pattern settings
    enable_trianglemix: bool = False
    num_triangle_layers: Optional[int] = None
    triangle_layer_indices: List[int] = field(default_factory=list)
    
    # Attention pattern parameters
    num_sink_tokens: int = 4
    sliding_window_size: int = 32
    num_last_tokens: int = 64
    
    # Ascend NPU optimization
    use_npu_optimization: bool = True
    npu_block_size: int = 64
    
    # Gradient-based analysis (for selecting Triangle layers)
    use_gradient_analysis: bool = False
    analysis_warmup_steps: int = 100
    
    def to_trianglemix_config(self) -> Optional[TriangleMixConfig]:
        """Convert to TriangleMixConfig."""
        if not self.enable_trianglemix:
            return None
        
        return TriangleMixConfig(
            num_sink_tokens=self.num_sink_tokens,
            sliding_window_size=self.sliding_window_size,
            num_last_tokens=self.num_last_tokens,
            num_triangle_layers=self.num_triangle_layers,
            triangle_layer_indices=self.triangle_layer_indices if self.triangle_layer_indices else None,
        )
    
    @staticmethod
    def for_ascend_npu(
        num_triangle_layers: int = 12,
        enable_gradient_analysis: bool = True,
    ) -> "TriangleMixInferenceConfig":
        """Create config optimized for Ascend NPU."""
        return TriangleMixInferenceConfig(
            enable_trianglemix=True,
            num_triangle_layers=num_triangle_layers,
            use_npu_optimization=True,
            use_gradient_analysis=enable_gradient_analysis,
        )
    
    @staticmethod
    def for_qwen3(
        model_size: str = "14B",
        num_triangle_layers: Optional[int] = None,
    ) -> "TriangleMixInferenceConfig":
        """Create config optimized for Qwen3 models."""
        if num_triangle_layers is None:
            # Default: apply TriangleMix to lower layers
            if model_size == "14B":
                num_triangle_layers = 10
            elif model_size == "32B":
                num_triangle_layers = 12
            else:
                num_triangle_layers = 8
        
        return TriangleMixInferenceConfig(
            enable_trianglemix=True,
            num_triangle_layers=num_triangle_layers,
            num_sink_tokens=4,
            sliding_window_size=32,
            num_last_tokens=64,
            use_npu_optimization=True,
        )


# Example usage documentation
EXAMPLE_USAGE = """
# Example 1: Basic TriangleMix setup for Ascend NPU
from vllm.attention.trianglemix_config import TriangleMixInferenceConfig

config = TriangleMixInferenceConfig.for_ascend_npu(num_triangle_layers=12)

# Example 2: Qwen3 specific configuration
config = TriangleMixInferenceConfig.for_qwen3(model_size="14B")

# Example 3: Custom layer selection
config = TriangleMixInferenceConfig(
    enable_trianglemix=True,
    triangle_layer_indices=[0, 1, 2, 3, 4, 5],  # Apply to first 6 layers
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
)

# Get TriangleMixConfig for use in model
trianglemix_cfg = config.to_trianglemix_config()

# Example 4: Environment variables for runtime configuration
# Set in your inference script or system environment:
# export VLLM_TRIANGLEMIX_ENABLED=1
# export VLLM_TRIANGLEMIX_LAYERS=12
# export VLLM_NPU_OPTIMIZATION=1
"""
