# 1. Dùng chunking để làm bot trả lời tiểu sử người nổi tiếng, anime v...v
#   - <https://en.wikipedia.org/wiki/S%C6%A1n_T%C3%B9ng_M-TP>
#   - <https://en.wikipedia.org/wiki/Jujutsu_Kaisen>

import chromadb
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
load_dotenv()

def get_wikipedia_text(topic: str) -> str:
    wiki = Wikipedia('thangnaozay', 'en')
    page = wiki.page(topic)
    return page.text if page.exists() else ""

query = "em trai của Sơn Tùng M-TP?" 
# Nếu hỏi mono là ai thì trả lời là em trai..., nhưng hỏi em của ST thì ẻm ko biết
topic = "Sơn Tùng M-TP"

text = get_wikipedia_text(topic)
if not text:
    print("Không tìm thấy thông tin trên Wikipedia.")
    import sys
    sys.exit()
    

COLLECTION_NAME = re.sub(r'[^a-zA-Z0-9_-]', '_', topic)  # Convert invalid characters to "_"
client = chromadb.PersistentClient(path="./data")   
client.heartbeat()

if COLLECTION_NAME in client.list_collections():
    client.delete_collection(COLLECTION_NAME)

embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
paraghraphs = text.split('\n\n')

for index, paraghraph in enumerate(paraghraphs):
    collection.add(documents=[paraghraph], ids=[str(index)])
    
q = collection.query(query_texts=[query], n_results=3)
CONTEXT = q["documents"][0]

# Xây dựng prompt, đưa dữ liệu vừa truy xuất vào làm ngữ cảnh trong biến `CONTEXT`
prompt = f"""
Use the following CONTEXT to answer the QUESTION at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use an unbiased and journalistic tone.

CONTEXT: {CONTEXT}

QUESTION: {query}
"""

# In ra prompt cuối cùng (lược bước phần context)


# https://platform.openai.com/api-keys
OPEN_API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=OPEN_API_KEY)

# Đặt câu hỏi thông thường không dùng RAG
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt},
    ]
)

print(response.choices[0].message.content)


