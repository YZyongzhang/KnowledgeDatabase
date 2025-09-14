# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer

model_dir = r"C:\Users\Admin\Desktop\KnowledgeDatabase\llm\Qwen\Qwen-4b"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

while True:
	s = input("input your ask ?")
	messages = [
		{"role": "user", "content": s},
	]
	inputs = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		tokenize=True,
		return_dict=True,
		return_tensors="pt",
	).to(model.device)
	streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True )
	outputs = model.generate(**inputs, max_new_tokens=4000 , streamer=streamer)