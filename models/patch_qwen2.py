from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from .rope_attn_fwd import attention_forward

def patch_attn():
    Qwen2Attention.forward = attention_forward
