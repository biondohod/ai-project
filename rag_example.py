# rag_example.py
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Инициализация токенизатора и модели
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# Тестовый запрос
input_text = "What is artificial intelligence?"
inputs = tokenizer([input_text], return_tensors="pt")
generated = model.generate(**inputs)
output = tokenizer.batch_decode(generated, skip_special_tokens=True)
print("Generated answer:", output[0])