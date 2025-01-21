from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from time import time
import torch

ROOT_DIR = Path(__file__).parent
cache_dir =  ROOT_DIR / ".cache" / "huggingface" / "hub"

prompt = "海内存知己的下一句是"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=cache_dir,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto",
    # attn_implementation="flash_attention_2",
    cache_dir=cache_dir,
)
print(model.config)
print(model)

model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
print(model_inputs)

start = time()
generation_output = model.generate(
    **model_inputs,
    max_new_tokens=50,
    return_dict_in_generate=True,
    repetition_penalty=1.2,
)
torch.cuda.synchronize()
end = time()
print("Elapsed time(s):", end - start)
responses = tokenizer.batch_decode(**generation_output, skip_special_tokens=True)
print(responses)


from models.patch_qwen2 import patch_attn
patch_attn()


# from models.modeling_qwen2 import Qwen2ForCausalLM
# model = Qwen2ForCausalLM.from_pretrained(
#     "Qwen/Qwen2.5-0.5B-Instruct",
#     # 用 device_map 或 low_cpu_mem_usage 可以避免内存不足，参考
#     # https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/model#large-model-loading
#     # Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`
#     device_map="auto",
#     torch_dtype="auto",
#     cache_dir=cache_dir,
# )


model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
print(model_inputs)
start = time()
generation_output = model.generate(
    **model_inputs,
    max_new_tokens=50,
    return_dict_in_generate=True,
    repetition_penalty=1.2,
)
torch.cuda.synchronize()
end = time()
print("Elapsed time(s):", end - start)
responses = tokenizer.batch_decode(**generation_output, skip_special_tokens=True)
print(responses)
