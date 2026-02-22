from transformers import AutoTokenizer, AutoModel

model_name = "Qwen/Qwen3-Embedding-0.6B"
save_path = "./models/embedding"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

model_name = "Qwen/Qwen3-Reranker-0.6B"
save_path = "./models/reranker"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
