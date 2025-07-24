import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset , DataLoader
from tokenizers import ByteLevelBPETokenizer


class QADataset(Dataset):
    def __init__(self, data, tokenizer, block_size=512,pad_token="<pad>"):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = tokenizer.token_to_id(pad_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = f"问：{example['instruction']}\n答：{example['output']}"
        tokens = self.tokenizer.encode(text)

        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens += [self.pad_token_id] * (self.block_size - len(tokens))  # 这里用手动获取的pad_token_id

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # 训练时忽略pad

        return {"input_ids": input_ids, "labels": labels}

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=128, n_layer=4, n_head=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size

        tok_emb = self.token_emb(idx)           # (B,T,C)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb                   # (B,T,C)

        # Transformer decoder layers
        for layer in self.layers:
            x = layer(x, x)  # 注意这里用decoder层同时做self-attention和交叉注意力（简化写法）

        x = self.ln_f(x)                        # (B,T,C)
        logits = self.head(x)                   # (B,T,vocab_size)

        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

model = SimpleGPT(vocab_size=614, block_size=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
num_epochs = 100
tokenizer = ByteLevelBPETokenizer(
    "tokenizer_output/vocab.json",
    "tokenizer_output/merges.txt",
)
data = [
    {"instruction": "路径", "output": "使用sys模块，然后使用sys.path.append()将路径导入"},
    {"instruction": "路径", "output": "使用os.path.dirname 返回不带本文件的文件目录"},
    {"instruction": "pip", "output": "pip的报错可以考虑由于依赖版本不对导致的，比如python版本，比如各种包的冲突"},
    {"instruction": "argparse 链接", "output": "https://blog.csdn.net/qq_43391414/article/details/120097139"},
    {"instruction": "argparse ipynb 报错", "output": "缺少一些必要参数"},
    {"instruction": "argparse ipynb 解决方案", "output": "在最后的封装中使用parser.parse_args(args=[])"},
    {"instruction": "argparse ipynb 报错原因", "output": "由于parse_args会从系统自动的读取参数，但是当我们使用命令行的时候这个参数是不会传输的，所以就默认是args = [] , 但是当我们使用ipynb的时候，ipynb会自动的传入一个路径字符串等类的，然后就会出现找不到这个参数。解决这个问题其实也很简单，两种方式，一种是将这个参数固定为null，也就是给出的这个形式，另一种就是在里面写一个字符串的参数接受ipynb里面的数据，然后就不会报错了。"},
    {"instruction": "术语 grandtruth 定义", "output": "标准真相"},
    {"instruction": "术语 grandtruth 应用", "output": "运用到人工智能领域的话可以认为是正确答案，比如说物体识别中的标注。"},
    {"instruction": "python 枚举 规范", "output": "枚举可以实现一个数据到另一个数据的映射。"},
    {"instruction": "python 枚举 代码", "output": "使用from enum import Enum 然后定义class your_class(Enum)之后在类中直接写 one = one two = two 。"},
    {"instruction": "python 枚举 使用", "output": "在我们定义好的类中不需要init 示例，可以直接使用类进行获取映射。your_class(one).name "},
    {"instruction": "python 枚举 注意", "output": "使用enum中，需要使用后面的匹配前面的，然后使用name获取到前面的。"},
    {"instruction": "python 注解 内容", "output": "@classmethod 注解这个方法为类方法而不是实例方法。"},
    {"instruction": "python 注解 核心", "output": "注解为类方法之后可以通过类直接进行调用，而不用实例化这个类，然后在调用里面的方法了。"},
    {"instruction": "python 注解 误解", "output": "类方法的第一个参数是cls 是class的缩写，而实例方法的第一个参数是self。"},
    {"instruction": "yolo 背景", "output": "在类人注意力机制中利用yolo的置信度进行计算reward"},
    {"instruction": "yolo 核心", "output": "分割出图像的物体，然后进行标注"},
    {"instruction": "yolo 代码", "output": "从from ultralytics import YOLO 然后初始化model, yolo_model = YOLO(\"yolo11n.pt\") ,之后直接可以使用yolo_model(img_path , save=true , save_path = '')sadfsaf:"},
    {"instruction": "mel 背景", "output": "在深度学习的过程中，对于图像来说，每一个像素的维度就是我们想要拟合训练的，但是对于音频来说，我们拿不到这种数据。"},
    {"instruction": "mel 方法", "output": "在传统的方式中，我们采用了语谱图，对于语谱图来说，将x轴当作时间坐标，将y轴当作频率，然后将xy对应的点，是这个时刻频率的幅度。但是这种方式特征有一定问题。因为我们是想要通过模拟人类而言，而对于人类的话，人类对声音的感知是对数的形式，比如在低频段敏感，对于高频段不敏感，然后就出现了mel语谱图"},
    {"instruction": "mel 技术", "output": "mel通过对于初始的f进行二次转化得到了一个符合我们预期的特征图"}
]

dataset = QADataset(data, tokenizer, block_size=512)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        targets = batch["labels"].to(device)

        logits, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} loss: {loss.item():.4f}")