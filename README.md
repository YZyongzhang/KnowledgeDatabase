

<h1 align="center">📚 KnowledgeDatabase 知识库项目</h1>

<p align="center">
  <b>一个基于 PyQt6 的简易图形化知识库管理工具</b><br>
  📝 轻量 · ⚡ 实时保存 · 💾 JSON 持久化
</p>

---

## ✨ 功能特点

- 🎨 图形化界面，基于 **PyQt6**
- 🔑 输入 `Key` 和 `Value`，快速添加数据
- 💾 数据以 JSON 格式存储，可持久化保存

---

## 📂 项目结构

```plaintext
KnowledgeDatabase/
│
├── Window/
│   └── win.py       # 主窗口逻辑 (PyQt6 界面)
├── DataImpl.py      # 数据写入/读取逻辑
├── main.py          # 程序入口
├── data.json        # 数据存储文件 (运行时生成/更新)
└── README.md        # 项目说明
````

---

## 🚀 使用方法

### 1. 安装依赖

确保你的环境已经安装了 Python 3.9+
然后安装 PyQt6：

```bash
pip install PyQt6
```

### 2. 运行程序

```bash
python main.py
```

### 3. 使用

* 在界面中输入 `Key` 和 `Value`
* 点击 **提交** 按钮
* 数据会自动写入 `data.json` 文件

---

## 📝 数据存储示例

```json
{
    "test": "test",
}
```

---

## 🔧 待实现/扩展

* [ ] 🔍 数据查询与搜索
* [ ] ✏️ 编辑和删除已有数据
* [ ] 📊 导出数据为 CSV/Excel
* [ ] 🎨 界面美化（支持主题切换）
* [ ] 👥 多用户/多知识库管理

---



