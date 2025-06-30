# Flash Attention CuTe

使用 CUTLASS CuTe 复现 Flash Attention 2 算子

## TODO

- [x] 实现 TensorCore gemm 计算
- [x] 实现 ldmatrix 加载数据
- [x] 实现 two-stage gemm pipeline
- [x] 实现 bank-conflict-free 共享内存读写
- [x] 支持 fp16 和 bf16 数据类型
- [x] 实现算子调度
- [x] 支持 causal mask
- [x] 支持 MHA, GQA, MQA
- [x] 实现 GQA 下 合并同组 q head，以降低冗余 TensorCore 计算
- [x] 实现 python 调用 和 pytorch 自定义算子集成
- [x] 实现 llm 注意力机制集成
- [ ] 支持 varlen
- [ ] 优化 mask 性能
- [ ] 实现 splitkv 算子（Flash Decoding）
- [ ] 实现 paged kv cache（Paged Attention）
- [ ] 支持量化数据类型的算子

## Requirements

```
torch
transformers
flash-attn
```

## Quick Start

WIP

## Performance

WIP

## Resources

WIP

## Acknowledgement

1. [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
2. [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
