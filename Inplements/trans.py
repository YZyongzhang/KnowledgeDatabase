import json
from typing import List, Tuple, Dict
import pdb
# ------- 加载 JSON 知识库 --------

def load_knowledge_json(json_path: str) -> Dict:
    pdb.set_trace()
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------- 核心函数部分 --------

def flatten_knowledge(kb: Dict, prefix: str = '') -> List[Tuple[str, str]]:
    """将嵌套字典 flatten 成 path 和 value 的列表"""
    items = []
    for key, val in kb.items():
        new_prefix = f"{prefix}/{key}".strip('/')
        if isinstance(val, dict):
            items.extend(flatten_knowledge(val, new_prefix))
        else:
            items.append((new_prefix, str(val)))
    return items

def format_qa(question: str, answer: str) -> str:
    return f"Q: {question}\nA: {answer}\n\n"

def format_instruction(question: str, answer: str) -> str:
    return f"### Instruction:\n{question}\n\n### Response:\n{answer}\n\n"

def format_jsonl_entry(question: str, answer: str) -> str:
    return json.dumps({"instruction": question, "output": answer}, ensure_ascii=False)

# ------- 主处理函数 --------

def generate_corpora(flat_data: List[Tuple[str, str]], out_prefix="knowledge_corpus"):
    with open(f"{out_prefix}_qa.txt", "w", encoding="utf-8") as f_qa, \
         open(f"{out_prefix}_instruction.txt", "w", encoding="utf-8") as f_inst, \
         open(f"{out_prefix}.jsonl", "w", encoding="utf-8") as f_jsonl:

        for path, value in flat_data:
            key = path.split('/')[-1]
            question = f"{key} 是什么意思？"
            answer = value.strip()
            f_qa.write(format_qa(question, answer))
            f_inst.write(format_instruction(question, answer))
            f_jsonl.write(format_jsonl_entry(question, answer) + "\n")

# ------- 执行入口 --------

if __name__ == "__main__":
    json_path = "./DataBase/raw_files/DataBase.yaml"  # ← 替换为你的知识库 JSON 路径
    knowledge = load_knowledge_json(json_path)
    flattened = flatten_knowledge(knowledge)
    generate_corpora(flattened)
    print("✅ 训练语料已生成：")
    print("  - knowledge_corpus_qa.txt")
    print("  - knowledge_corpus_instruction.txt")
    print("  - knowledge_corpus.jsonl")
