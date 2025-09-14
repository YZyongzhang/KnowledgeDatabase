from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# 1. 加载模型
model_dir = r"C:\Users\Admin\Desktop\KnowledgeDatabase\llm\Qwen\Qwen-4b"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", dtype=torch.float16)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,                     # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # 可微调的模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

"""
{
  "prompt": "用户输入或任务描述",
  "completion": "模型期望输出"
}

"""
# 3. 准备数据集
dataset = load_dataset("json", data_files="./data/dataset.jsonl")  # 你的微调数据

IGNORE_INDEX = -100

def tokenize_function(examples):
    # 编码 prompt
    prompt_encodings = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=256)
    # 编码 completion
    completion_encodings = tokenizer(examples["completion"], truncation=True, padding="max_length", max_length=256)

    input_ids = []
    labels = []
    for p_ids, c_ids in zip(prompt_encodings["input_ids"], completion_encodings["input_ids"]):
        # 输入 = prompt + completion
        ids = p_ids + c_ids[1:]   # 去掉 completion 的起始token
        input_ids.append(ids)

        # 标签：prompt 部分设为 -100（忽略），只训练 completion 部分
        label = [IGNORE_INDEX] * len(p_ids) + c_ids[1:]
        labels.append(label)

    return {
        "input_ids": input_ids,
        "attention_mask": [[1]*len(ids) for ids in input_ids],
        "labels": labels
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir="./qwen-4b-lora",
    per_device_train_batch_size=2,
    num_train_epochs=30,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    report_to="tensorboard",
    save_strategy="epoch",
)

# 5. Trainer 微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
