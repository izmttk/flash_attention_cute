from transformers.models.llama.modeling_llama import LlamaAttention
from .rope_attn_fwd import attention_forward

def patch_attn():
    LlamaAttention.forward = attention_forward
