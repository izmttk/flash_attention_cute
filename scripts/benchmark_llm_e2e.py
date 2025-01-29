import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import models as model_hub
from time import time

import argparse
import sys
from pathlib import Path

prompt_template = r"""
Imagine you are a highly advanced AI system tasked with generating a comprehensive report on the future of artificial intelligence and its impact on various sectors of society. Your report should cover the following areas in detail: 
1. **Introduction to Artificial Intelligence**: Provide a brief history of AI, starting from its inception in the 1950s to the present day. Discuss key milestones, such as the development of neural networks, the advent of machine learning, and the rise of deep learning. Explain how AI has evolved over the decades and what has driven its rapid advancement in recent years.
2. **Current State of AI**: Analyze the current state of AI technology. Discuss the most significant breakthroughs in AI research, including natural language processing, computer vision, and reinforcement learning. Provide examples of real-world applications of AI in industries such as healthcare, finance, transportation, and entertainment. Highlight the challenges that AI currently faces, including issues related to bias, transparency, and ethical concerns.
3. **Future Trends in AI**: Predict the future trends in AI over the next 20 years. Discuss the potential for advancements in areas such as quantum computing, autonomous systems, and AI-driven creativity. Explore how AI might evolve to become more general-purpose, potentially leading to the development of Artificial General Intelligence (AGI). Consider the implications of AGI on society, including both the opportunities and risks it presents.
4. **Economic Impact of AI**: Examine the economic impact of AI on global markets. Discuss how AI is expected to transform industries, create new job opportunities, and disrupt existing business models. Analyze the potential for AI to drive economic growth, increase productivity, and reduce costs. Consider the potential for job displacement and the need for workforce reskilling in an AI-driven economy.
5. **Social and Ethical Implications**: Explore the social and ethical implications of widespread AI adoption. Discuss issues such as privacy, surveillance, and the potential for AI to be used in harmful ways. Consider the ethical responsibilities of AI developers and the need for regulations to ensure that AI is used in a way that benefits society as a whole. Discuss the potential for AI to exacerbate existing social inequalities and what can be done to mitigate these risks.
6. **AI and the Environment**: Analyze the environmental impact of AI. Discuss the energy consumption of large-scale AI models and data centers, and explore ways to make AI more sustainable. Consider how AI can be used to address environmental challenges, such as climate change, by optimizing energy usage, improving resource management, and supporting conservation efforts.
7. **AI in Education and Research**: Discuss the role of AI in transforming education and research. Explore how AI can be used to personalize learning experiences, improve educational outcomes, and support lifelong learning. Consider the potential for AI to accelerate scientific research by analyzing large datasets, generating hypotheses, and automating experiments.
8. **AI and Global Governance**: Examine the role of AI in global governance and international relations. Discuss how AI can be used to enhance decision-making, improve public services, and support international cooperation. Consider the challenges of regulating AI on a global scale and the need for international agreements to ensure that AI is developed and used responsibly.
9. **Conclusion**: Summarize the key points of your report and provide a forward-looking perspective on the future of AI. Discuss the importance of continued research, collaboration, and ethical considerations in shaping the future of AI. Emphasize the need for a balanced approach that maximizes the benefits of AI while minimizing its risks.
Your report should be well-structured, with clear headings and subheadings for each section. Use data, statistics, and examples to support your arguments. Ensure that your writing is clear, concise, and free of jargon, making it accessible to a broad audience. Finally, include a list of references to credible sources that you used to inform your analysis.
""".strip()


ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT.resolve()))

def run_perf(
    model,
    tokenizer,
    prompt="Hello, how are you?",
    max_new_tokens=200,
):
    
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
        
    torch.cuda.synchronize()
    start = time()
    generation_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        # repetition_penalty=1.2,
    )
    torch.cuda.synchronize()
    end = time()
    elapsed_time = end - start
    
    output_ids = generation_output.sequences[:, seq_len:]
    new_tokens = output_ids.shape[1]
    
    throughput = new_tokens / elapsed_time
    print(f"[Generate] Throughput: {throughput:.2f} tokens/s | SeqLen: {seq_len} | NewTokens: {new_tokens} | ElapsedTime: {elapsed_time:.4f}s")
    
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(responses)

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM prefilling/decoding throughput")
    parser.add_argument("--model", type=str, required=True,
                      help="HuggingFace 模型名称或路径")
    parser.add_argument("--attn", type=str, default=None,
                      help="Attention 实现")
    parser.add_argument("--prompt", type=str, default=prompt_template,
                      help="输入文本提示")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                      help="解码阶段生成的最大 token 数量")
    parser.add_argument("--torch-dtype", type=str, default="auto",
                      help="模型计算精度")
    parser.add_argument("--device", type=str, default="auto",
                      help="模型推理设备")
    args = parser.parse_args()
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device,
        attn_implementation=None if args.attn == "custom" else args.attn,
    ).eval()
    
    if (args.attn == "custom"):
        if isinstance(model, model_hub.qwen2.Qwen2ForCausalLM):
            from models.patch_qwen2 import patch_attn
        elif isinstance(model, model_hub.llama.LlamaForCausalLM):
            from models.patch_llama import patch_attn
        else:
            raise RuntimeError(f"Model {args.model} is not supported currently under custom attention implementation.")
        patch_attn()
    
    run_perf(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
    )
    
if __name__ == "__main__":
    main()
    
"""
Usage Examples:

using default model config:
python benchmark_llm_e2e.py --model "meta-llama/Llama-2-7b-chat-hf"

using project's flash attention implementation:
python benchmark_llm_e2e.py --model "meta-llama/Llama-2-7b-chat-hf" --attn "custom"

full arguments call:
python benchmark_llm_e2e.py \
  --model "mmeta-llama/Llama-2-7b-chat-hf" \
  --attn custom \
  --prompt "The meaning of life is" \
  --torch-dtype bfloat16 \
  --device auto
"""
