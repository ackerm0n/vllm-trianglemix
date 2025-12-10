# TriangleMix Attention Implementation Guide

## 概述

TriangleMix是一种高效的静态稀疏注意力模式，专门为长上下文预填充优化。该实现在vLLM框架中集成了TriangleMix机制，支持Ascend NPU上的Qwen3模型推理。

## 核心概念

### 1. 三个注意力区域划分

基于位置(i, j)对注意力矩阵进行划分：

- **Streaming Section（流处理部分）**
  - 包含注意力汇聚（Attention Sink）和滑动窗口
  - 掩码条件：`i >= j and (j <= si or i - j <= sl)`
  - 其中`si`=注意力汇聚数，`sl`=滑动窗口大小

- **Last Q-K Section（最后查询-键部分）**
  - 覆盖序列最后部分的Q-K交互
  - 掩码条件：`i >= j and N - i < last and j > si and i - j > sl`
  - 其中`N`=序列长度，`last`=最后部分行数

- **Middle Q-K Section（中间查询-键部分）**
  - 包含中间部分的Q-K交互
  - 掩码条件：`i >= j and N - i >= last and j > si and i - j > sl`
  - **该部分在选中的层被跳过**

### 2. Triangle注意力模式

Triangle注意力 = Streaming + Last（不包含Middle）

- 将复杂度从O(N²)降低到O(N)
- 完全静态，无需动态稀疏预测
- 实现简单高效

## 实现架构

### 文件结构

```
vllm/attention/
├── trianglemix_attention.py      # 核心TriangleMix实现
├── trianglemix_config.py         # 配置定义
├── ascend_trianglemix.py         # Ascend NPU优化
└── __init__.py

vllm/model_executor/models/
└── qwen3.py                      # Qwen3模型集成

tests/
└── trianglemix_attention_test.py  # 测试套件
```

### 核心模块

#### 1. `TriangleMixConfig` - 配置类

```python
TriangleMixConfig(
    num_sink_tokens: int = 4,           # 注意力汇聚数
    sliding_window_size: int = 32,      # 滑动窗口大小
    num_last_tokens: int = 64,          # 最后部分的行数
    num_triangle_layers: Optional[int] = None,  # 应用Triangle的层数
    triangle_layer_indices: Optional[list[int]] = None,  # 具体层索引
)
```

#### 2. `TriangleMixMaskGenerator` - 掩码生成器

```python
generator = TriangleMixMaskGenerator(config)

# 生成各类掩码
streaming_mask = generator._create_streaming_mask(seq_len, device, dtype)
last_mask = generator._create_last_mask(seq_len, device, dtype)
middle_mask = generator._create_middle_mask(seq_len, device, dtype)
triangle_mask = generator._create_triangle_mask(seq_len, device, dtype)
dense_mask = generator._create_dense_mask(seq_len, device, dtype)

# 获取attention掩码（0/-inf格式）
attn_mask = generator.get_attention_mask(seq_len, layer_idx, device, dtype, use_triangle)
```

#### 3. `TriangleMixAttention` - 高级接口

```python
attn = TriangleMixAttention(config, num_layers=32)

# 检查某层是否使用Triangle
use_triangle = attn.mask_generator.should_use_triangle(layer_idx)

# 获取该层的attention掩码
mask = attn.get_attn_mask(seq_len, layer_idx, device, dtype)
```

#### 4. `TriangleMixAnalyzer` - 梯度分析

```python
analyzer = TriangleMixAnalyzer(num_layers=32)

# 记录Middle Q-K的梯度贡献
analyzer.record_middle_qk_gradient(layer_idx, gradient)

# 获取应该应用Triangle的层（贡献度最低的层）
triangle_layers = analyzer.get_triangle_layers(num_triangle_layers=12)
```

#### 5. `AscendTriangleMixAttention` - NPU优化

```python
ascend_attn = AscendTriangleMixAttention(
    num_heads=32,
    head_dim=128,
    scale=1.0/sqrt(128),
    trianglemix_config=config,
    layer_idx=layer_idx,
    num_layers=num_layers,
)

# 计算attention（自动应用TriangleMix掩码）
output = ascend_attn(query, key, value)
```

## 使用方法

### 方案1：在Qwen3模型中使用

```python
from vllm.attention.trianglemix_config import TriangleMixInferenceConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

# 创建TriangleMix配置
trianglemix_config = TriangleMixInferenceConfig.for_qwen3(
    model_size="14B",
    num_triangle_layers=12,  # 在前12层应用Triangle模式
).to_trianglemix_config()

# 在模型初始化时传递配置
# （需要修改Qwen3Model.__init__来传递此配置到各层）
```

### 方案2：独立的掩码生成

```python
from vllm.attention.trianglemix_attention import TriangleMixConfig, TriangleMixMaskGenerator

config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    num_triangle_layers=12,
)

generator = TriangleMixMaskGenerator(config)

# 获取某一层的掩码
mask = generator.get_attention_mask(
    seq_len=2048,
    layer_idx=5,
    device=torch.device('cuda'),
    dtype=torch.float32,
    use_triangle=True,  # 该层使用Triangle
)
```

### 方案3：Ascend NPU优化

```python
from vllm.attention.ascend_trianglemix import (
    AscendTriangleMixAttention,
    AscendNPUTriangleMixOptimizer,
)

# 创建NPU优化的attention
attn = AscendTriangleMixAttention(
    num_heads=32,
    head_dim=128,
    scale=1.0/sqrt(128),
    trianglemix_config=config,
)

# 优化掩码
optimized_mask = AscendNPUTriangleMixOptimizer.optimize_mask_for_npu(mask)

# 执行attention
output = attn(query, key, value, attn_mask=optimized_mask)
```

## 性能特性

### 复杂度分析

| 模式 | Q-K复杂度 | 注意力复杂度 | 内存需求 |
|------|---------|----------|--------|
| Dense | O(N²) | O(N²) | O(N²) |
| Triangle | O(si·N + sl·N + last·N) | O(N) | O(N) |
| 压缩比 | ~1/(2-4)x | ~1/(2-4)x | ~1/(2-4)x |

### 优化策略

1. **静态模式**：不需要运行时预测或动态稀疏索引
2. **NPU友好**：块级操作适合Ascend NPU执行
3. **梯度分析**：自动识别可以转换的层
4. **可配置**：灵活的参数调整

## 参数指南

### Qwen3 14B推荐配置

```python
TriangleMixConfig(
    num_sink_tokens=4,          # 保持4个汇聚令牌
    sliding_window_size=32,     # 32的滑动窗口
    num_last_tokens=64,         # 64个最后令牌
    num_triangle_layers=12,     # 前12层（共32层中）使用Triangle
)
```

### Qwen3 32B推荐配置

```python
TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    num_triangle_layers=16,     # 前16层（共64层中）使用Triangle
)
```

### 长上下文推荐配置

对于超长序列（>4096）：

```python
TriangleMixConfig(
    num_sink_tokens=8,          # 更多汇聚令牌
    sliding_window_size=64,     # 更大窗口
    num_last_tokens=128,        # 更多最后令牌
    num_triangle_layers=16,     # 更多Triangle层
)
```

## 梯度分析工作流

### 1. 收集梯度

```python
analyzer = TriangleMixAnalyzer(num_layers=32)

# 在前向传播中记录各层的Middle Q-K梯度
for layer_idx, gradient in enumerate(layer_gradients):
    analyzer.record_middle_qk_gradient(layer_idx, gradient)
```

### 2. 选择Triangle层

```python
# 选择贡献度最低的12层
triangle_layers = analyzer.get_triangle_layers(num_triangle_layers=12)
# 结果：[12, 13, 14, ..., 23]  (通常是较高的层)
```

### 3. 应用配置

```python
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    triangle_layer_indices=triangle_layers,
)
```

## 测试与验证

### 运行测试套件

```bash
python tests/trianglemix_attention_test.py
```

测试包括：
- 掩码生成正确性
- Triangle vs Dense对比
- 梯度分析
- NPU优化
- 性能基准

### 验证掩码正确性

```python
from tests.trianglemix_attention_test import test_streaming_mask, test_triangle_vs_dense

test_streaming_mask()
test_triangle_vs_dense()
```

## Ascend NPU集成建议

### 1. 使用块级操作

```python
block_size = 64  # Ascend NPU推荐块大小
optimizer = AscendNPUTriangleMixOptimizer()
optimized_mask = optimizer.optimize_mask_for_npu(mask, block_size=block_size)
```

### 2. 避免-inf值

```python
# Ascend NPU对-inf处理性能不佳，自动转换为大负数
mask = optimizer.optimize_mask_for_npu(mask)  # -inf → -1e6
```

### 3. 块级稀疏计算

```python
# 对于大型序列，使用块级稀疏attention
output = optimizer.sparse_attention_kernel(
    query, key, value, mask, scale=scale, block_size=64
)
```

## 常见问题

### Q1: 如何选择num_triangle_layers？

根据模型大小和目标性能：
- 小型模型（7B）：6-8层
- 中型模型（14B）：10-12层
- 大型模型（32B+）：16-20层

建议使用梯度分析来自动选择。

### Q2: num_last_tokens应该设为多少？

一般设为`sliding_window_size * 2`：
- 如果`sliding_window_size=32`，设`num_last_tokens=64`
- 确保"最后"部分覆盖关键令牌

### Q3: Triangle模式对准确率有影响吗？

实验表明，对合理选择的层应用Triangle不会显著影响准确率，因为这些层的Middle Q-K贡献本来就很小。

### Q4: 如何在动态批处理中应用TriangleMix？

当序列长度不同时：
- 动态生成对应长度的掩码
- 掩码缓存可自动处理多个序列长度
- 调用`clear_cache()`重置缓存

## 性能评估

### 推荐的评估指标

1. **延迟改进**：与Dense相比的推理延迟降低
2. **吞吐量增益**：每秒处理的令牌数
3. **内存节省**：KV缓存内存使用
4. **准确率保留**：相对于Dense的准确率差异

### 基准命令

```bash
# 测试掩码创建性能
python tests/trianglemix_attention_test.py benchmark_mask_creation

# 测试完整推理
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B-Chat \
    --enable-trianglemix \
    --trianglemix-layers 12
```

## 参考实现

### Qwen3集成示例

```python
# vllm/model_executor/models/qwen3.py中的修改

from vllm.attention.trianglemix_config import TriangleMixInferenceConfig

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        # 从配置获取TriangleMix设置
        trianglemix_cfg = TriangleMixInferenceConfig.for_qwen3(
            model_size="14B"
        ).to_trianglemix_config()
        
        # 传递给model
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            trianglemix_config=trianglemix_cfg,
        )
```

## 故障排除

### 问题1: 掩码中出现NaN

**原因**：-inf值的softmax计算

**解决方案**：
```python
mask = optimizer.optimize_mask_for_npu(mask)  # 转换-inf为-1e6
```

### 问题2: Ascend NPU内存不足

**原因**：掩码矩阵过大

**解决方案**：
- 使用更小的`num_last_tokens`
- 应用更多层的Triangle模式
- 使用块级稀疏计算

### 问题3: 准确率下降

**原因**：选择了不合适的Triangle层

**解决方案**：
- 使用梯度分析自动选择
- 减少`num_triangle_layers`
- 调整`num_last_tokens`

## 参考论文

本实现基于TriangleMix论文，该论文提出：
- 三部分注意力区域划分
- Middle Q-K梯度贡献分析
- 静态Triangle注意力模式
- 与其他稀疏模式的对比

## 许可证

SPDX-License-Identifier: Apache-2.0

## 贡献指南

欢迎改进TriangleMix实现：
1. 优化Ascend NPU执行
2. 添加更多架构支持
3. 改进梯度分析算法
4. 性能优化
