from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers import Tokenizer
from tokenizers import ByteLevelBPETokenizer

# 训练一个 ByteLevel BPE 分词器（GPT2同款）
def train_tokenizer(files, vocab_size=1000, save_dir="tokenizer_output"):
    tokenizer = ByteLevelBPETokenizer()
    
    # 训练
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    
    # 保存
    tokenizer.save_model(save_dir)
    print(f"✅ 分词器已保存到: {save_dir}/")

# 调用
train_tokenizer(["knowledge_corpus_qa.txt"], vocab_size=1000)