# TriangleMix API 参考文档

## 目录
1. [TriangleMixConfig](#trianglemixconfig)
2. [TriangleMixMaskGenerator](#trianglemixmaskgenerator)
3. [TriangleMixAttention](#trianglemixattention)
4. [TriangleMixAnalyzer](#trianglemixanalyzer)
5. [AscendTriangleMixAttention](#ascendtrianglemixattention)
6. [AscendNPUTriangleMixOptimizer](#ascendnputrianglemixoptimizer)
7. [TriangleMixInferenceConfig](#trianglemixinferenceconfig)

---

## TriangleMixConfig

配置TriangleMix注意力模式的参数。

### 初始化

```python
config = TriangleMixConfig(
    num_sink_tokens: int = 4,
    sliding_window_size: int = 32,
    num_last_tokens: int = 64,
    num_triangle_layers: Optional[int] = None,
    triangle_layer_indices: Optional[list[int]] = None,
)
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_sink_tokens` | int | 4 | 注意力汇聚令牌数 |
| `sliding_window_size` | int | 32 | 滑动窗口大小 |
| `num_last_tokens` | int | 64 | Last Q-K部分的行数 |
| `num_triangle_layers` | int/None | None | 应用Triangle的层数（前N层） |
| `triangle_layer_indices` | list/None | None | 具体应用Triangle的层索引列表 |

### 属性

所有参数都直接作为属性访问。

### 示例

```python
# Qwen3-14B配置
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    num_triangle_layers=12,
)

# 长上下文配置
config = TriangleMixConfig(
    num_sink_tokens=8,
    sliding_window_size=64,
    num_last_tokens=128,
    num_triangle_layers=16,
)
```

---

## TriangleMixMaskGenerator

生成和管理TriangleMix注意力掩码。

### 初始化

```python
generator = TriangleMixMaskGenerator(config: TriangleMixConfig)
```

### 方法

#### `_create_streaming_mask(seq_len, device, dtype) -> Tensor`

创建Streaming掩码（注意力汇聚+滑动窗口）。

**参数：**
- `seq_len` (int): 序列长度
- `device` (torch.device): 张量设备
- `dtype` (torch.dtype): 张量数据类型

**返回：** 形状为(seq_len, seq_len)的掩码张量，值为0或1

**时间复杂度：** O(seq_len²)

#### `_create_last_mask(seq_len, device, dtype) -> Tensor`

创建Last Q-K掩码。

**参数：** 同上

**返回：** 形状为(seq_len, seq_len)的掩码张量

#### `_create_middle_mask(seq_len, device, dtype) -> Tensor`

创建Middle Q-K掩码。

**参数：** 同上

**返回：** 形状为(seq_len, seq_len)的掩码张量

#### `_create_triangle_mask(seq_len, device, dtype) -> Tensor`

创建Triangle掩码（Streaming + Last）。

**参数：** 同上

**返回：** 形状为(seq_len, seq_len)的掩码张量

#### `_create_dense_mask(seq_len, device, dtype) -> Tensor`

创建Dense（因果）掩码。

**参数：** 同上

**返回：** 形状为(seq_len, seq_len)的掩码张量（下三角）

#### `get_attention_mask(seq_len, layer_idx, device, dtype, use_triangle=False) -> Tensor`

获取用于注意力计算的掩码（0/-inf格式）。

**参数：**
- `seq_len` (int): 序列长度
- `layer_idx` (int): 层索引
- `device` (torch.device): 张量设备
- `dtype` (torch.dtype): 张量数据类型
- `use_triangle` (bool): 是否使用Triangle模式

**返回：** 形状为(seq_len, seq_len)的掩码（0表示允许，-inf表示掩码）

**缓存：** 自动缓存掩码以提高性能

#### `should_use_triangle(layer_idx: int) -> bool`

检查指定层是否应使用Triangle模式。

**参数：**
- `layer_idx` (int): 层索引

**返回：** bool，True表示使用Triangle

#### `clear_cache() -> None`

清除掩码缓存。

### 示例

```python
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=8,
    num_last_tokens=16,
)
generator = TriangleMixMaskGenerator(config)

# 生成Triangle掩码
mask = generator.get_attention_mask(
    seq_len=1024,
    layer_idx=0,
    device=torch.device('cuda'),
    dtype=torch.float32,
    use_triangle=True,
)

# mask中，0表示参与注意力，-inf表示被掩码
attended = (mask == 0).sum().item()
masked = (mask == float('-inf')).sum().item()
```

---

## TriangleMixAttention

高级接口，管理多层的Triangle注意力选择。

### 初始化

```python
attn = TriangleMixAttention(
    config: TriangleMixConfig,
    num_layers: int,
)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | TriangleMixConfig | 配置对象 |
| `num_layers` | int | 模型总层数 |

### 方法

#### `get_attn_mask(seq_len, layer_idx, device, dtype=torch.float32) -> Optional[Tensor]`

获取指定层的注意力掩码。

**参数：**
- `seq_len` (int): 序列长度
- `layer_idx` (int): 层索引
- `device` (torch.device): 张量设备
- `dtype` (torch.dtype): 张量数据类型

**返回：** 
- 掩码张量（如果seq_len > 2048且该层需要掩码）
- None（如果使用Dense注意力）

**特殊行为：** 短序列(<=2048)始终返回None，使用Dense注意力

#### `should_use_triangle(layer_idx) -> bool`

检查层是否使用Triangle。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `config` | TriangleMixConfig | 配置对象 |
| `num_layers` | int | 总层数 |
| `mask_generator` | TriangleMixMaskGenerator | 掩码生成器 |

### 示例

```python
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    num_triangle_layers=12,
)

attn = TriangleMixAttention(config=config, num_layers=32)

# 检查各层
for layer_idx in range(32):
    use_tri = attn.mask_generator.should_use_triangle(layer_idx)
    print(f"Layer {layer_idx}: {'Triangle' if use_tri else 'Dense'}")

# 获取某层的掩码
mask = attn.get_attn_mask(
    seq_len=2048,
    layer_idx=5,
    device=torch.device('cuda'),
)
```

---

## TriangleMixAnalyzer

基于梯度分析选择应用Triangle的层。

### 初始化

```python
analyzer = TriangleMixAnalyzer(num_layers: int)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_layers` | int | 模型总层数 |

### 方法

#### `record_middle_qk_gradient(layer_idx, gradient) -> None`

记录Middle Q-K部分的梯度贡献。

**参数：**
- `layer_idx` (int): 层索引
- `gradient` (Tensor): 梯度张量

**说明：** 应在反向传播中调用，可多次记录同一层

#### `get_triangle_layers(num_triangle_layers) -> List[int]`

基于记录的梯度获取应应用Triangle的层。

**参数：**
- `num_triangle_layers` (int): 要选择的层数

**返回：** 层索引列表，按升序排列

**算法：** 选择Middle Q-K贡献度最低的N层

#### `reset() -> None`

重置所有记录的梯度。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `num_layers` | int | 总层数 |
| `middle_qk_gradients` | List[List[float]] | 记录的梯度 |

### 示例

```python
analyzer = TriangleMixAnalyzer(num_layers=32)

# 在训练中记录梯度
for epoch in range(num_epochs):
    for layer_idx, layer in enumerate(model.layers):
        # 计算Middle Q-K的梯度
        grad = compute_gradient(layer.middle_qk)
        analyzer.record_middle_qk_gradient(layer_idx, grad)

# 获取最优的Triangle层
triangle_layers = analyzer.get_triangle_layers(num_triangle_layers=12)
print(f"Selected layers: {triangle_layers}")  # [20, 21, 22, ...]

# 创建配置
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    triangle_layer_indices=triangle_layers,
)
```

---

## AscendTriangleMixAttention

Ascend NPU优化的TriangleMix注意力实现。

### 初始化

```python
attn = AscendTriangleMixAttention(
    num_heads: int,
    head_dim: int,
    scale: float,
    trianglemix_config: Optional[TriangleMixConfig] = None,
    layer_idx: Optional[int] = None,
    num_layers: Optional[int] = None,
)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_heads` | int | 注意力头数 |
| `head_dim` | int | 每个头的维度 |
| `scale` | float | 注意力缩放因子（通常为1/√head_dim） |
| `trianglemix_config` | TriangleMixConfig/None | TriangleMix配置 |
| `layer_idx` | int/None | 当前层索引 |
| `num_layers` | int/None | 总层数 |

### 方法

#### `forward(query, key, value, attn_mask=None) -> Tensor`

计算注意力输出。

**参数：**
- `query` (Tensor): 查询张量，形状(batch_size, seq_len, num_heads, head_dim)
- `key` (Tensor): 键张量，形状(batch_size, seq_len, num_heads, head_dim)
- `value` (Tensor): 值张量，形状(batch_size, seq_len, num_heads, head_dim)
- `attn_mask` (Tensor/None): 可选的注意力掩码

**返回：** 注意力输出张量，形状(batch_size, seq_len, num_heads, head_dim)

**特性：** 自动应用TriangleMix掩码（如果配置）

### 示例

```python
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    num_triangle_layers=12,
)

attn = AscendTriangleMixAttention(
    num_heads=32,
    head_dim=128,
    scale=1.0 / (128 ** 0.5),
    trianglemix_config=config,
    layer_idx=5,
    num_layers=32,
)

# 前向传播
output = attn(query, key, value)  # 自动应用Triangle掩码
```

---

## AscendNPUTriangleMixOptimizer

Ascend NPU优化工具。

### 静态方法

#### `optimize_mask_for_npu(mask: Tensor, block_size: int = 64) -> Tensor`

优化掩码以提高NPU兼容性。

**参数：**
- `mask` (Tensor): 原始掩码（可能包含-inf）
- `block_size` (int): NPU块大小

**返回：** 优化后的掩码（-inf被转换为-1e6）

**处理：** 
- -inf (掩码值) → -1e6
- 0 (允许值) → 0

#### `sparse_attention_kernel(query, key, value, mask, scale, block_size=64) -> Tensor`

块级稀疏注意力计算。

**参数：**
- `query` (Tensor): 查询张量
- `key` (Tensor): 键张量
- `value` (Tensor): 值张量
- `mask` (Tensor): 稀疏掩码
- `scale` (float): 缩放因子
- `block_size` (int): 块大小

**返回：** 注意力输出

### 示例

```python
optimizer = AscendNPUTriangleMixOptimizer()

# 优化掩码
original_mask = generator.get_attention_mask(seq_len, layer_idx, device, dtype)
optimized_mask = optimizer.optimize_mask_for_npu(original_mask)

# 块级稀疏注意力
output = optimizer.sparse_attention_kernel(
    query, key, value, optimized_mask, scale=1.0/128, block_size=64
)
```

---

## TriangleMixInferenceConfig

推理时的TriangleMix配置（高级接口）。

### 初始化

```python
config = TriangleMixInferenceConfig(
    enable_trianglemix: bool = False,
    num_triangle_layers: Optional[int] = None,
    triangle_layer_indices: List[int] = [],
    num_sink_tokens: int = 4,
    sliding_window_size: int = 32,
    num_last_tokens: int = 64,
    use_npu_optimization: bool = True,
    npu_block_size: int = 64,
    use_gradient_analysis: bool = False,
    analysis_warmup_steps: int = 100,
)
```

### 静态工厂方法

#### `for_ascend_npu(num_triangle_layers=12, enable_gradient_analysis=False) -> TriangleMixInferenceConfig`

为Ascend NPU创建优化配置。

#### `for_qwen3(model_size="14B", num_triangle_layers=None) -> TriangleMixInferenceConfig`

为Qwen3模型创建配置。

**参数：**
- `model_size` (str): 模型大小 ("14B", "32B", ...)
- `num_triangle_layers` (int/None): 要覆盖的层数

### 方法

#### `to_trianglemix_config() -> Optional[TriangleMixConfig]`

转换为TriangleMixConfig。

**返回：** TriangleMixConfig或None（如果未启用）

### 示例

```python
# 快速创建Qwen3配置
config = TriangleMixInferenceConfig.for_qwen3(model_size="14B")
trianglemix_cfg = config.to_trianglemix_config()

# Ascend NPU配置
config = TriangleMixInferenceConfig.for_ascend_npu(
    num_triangle_layers=12,
    enable_gradient_analysis=True,
)

# 完全自定义
config = TriangleMixInferenceConfig(
    enable_trianglemix=True,
    triangle_layer_indices=[0, 1, 2, 3, 4, 5],
    num_sink_tokens=8,
    sliding_window_size=64,
    num_last_tokens=128,
    use_npu_optimization=True,
)
```

---

## 完整使用示例

### 示例1: 基本使用

```python
import torch
from vllm.attention.trianglemix_attention import TriangleMixConfig, TriangleMixMaskGenerator

# 创建配置
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    num_triangle_layers=12,
)

# 创建生成器
generator = TriangleMixMaskGenerator(config)

# 生成掩码
seq_len = 2048
mask = generator.get_attention_mask(
    seq_len=seq_len,
    layer_idx=0,
    device=torch.device('cuda'),
    dtype=torch.float32,
    use_triangle=True,
)

# 计算注意力
scores = torch.matmul(query, key.transpose(-2, -1)) * scale
scores = scores + mask
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, value)
```

### 示例2: 梯度分析

```python
from vllm.attention.trianglemix_attention import TriangleMixAnalyzer

# 创建分析器
analyzer = TriangleMixAnalyzer(num_layers=32)

# 记录梯度
for epoch in range(10):
    for layer_idx in range(32):
        grad = compute_layer_gradient(layer_idx)
        analyzer.record_middle_qk_gradient(layer_idx, grad)

# 获取最优层
triangle_layers = analyzer.get_triangle_layers(num_triangle_layers=12)

# 创建配置
config = TriangleMixConfig(
    num_sink_tokens=4,
    sliding_window_size=32,
    num_last_tokens=64,
    triangle_layer_indices=triangle_layers,
)
```

---

## 性能建议

| 参数 | 推荐值 | 范围 |
|------|--------|------|
| num_sink_tokens | 4-8 | 2-16 |
| sliding_window_size | 32-64 | 8-128 |
| num_last_tokens | 64-128 | 32-256 |
| num_triangle_layers | 12 (14B), 16 (32B) | 4-20 |
| npu_block_size | 64 | 32-256 |

---

## 错误处理

所有方法都会在参数无效时抛出异常：

```python
try:
    mask = generator.get_attention_mask(seq_len, layer_idx, device, dtype)
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

常见错误：
- `seq_len <= 0`: 无效的序列长度
- `layer_idx < 0` or `layer_idx >= num_layers`: 层索引超出范围
- 设备/dtype不兼容
