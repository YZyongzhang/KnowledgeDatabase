from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch

# 1. 基础模型 & LoRA 权重路径
base_model = r"C:\Users\Admin\Desktop\KnowledgeDatabase\llm\Qwen\Qwen-4b"
lora_model = r"./qwen-4b-lora/checkpoint-3390/"   # 你训练好的LoRA模型目录

# 2. 加载 tokenizer 和基础模型
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    dtype=torch.float16
)

# 3. 加载 LoRA 权重
model = PeftModel.from_pretrained(model, lora_model)

# 4. （可选）合并 LoRA 权重，推理更快
# model = model.merge_and_unload()

# 5. 定义 streamer（实时输出）
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

while True:
    prompt = input("input:")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer
        )
