from milvus.milvus import MilvusConnection
import json

milvus = MilvusConnection('/Users/mintel/Documents/milvus-rag/rag/milvus_rag_agent/milvus_demo.db')
print(milvus.list_collections())
text_to_embedded = milvus.read_and_split('/Users/mintel/Documents/milvus-rag/rag/milvus_rag_agent/milvus/faq/*.md')
milvus.insert_vectors(text_to_embedded[:10], 'test_rag')
kk = milvus.search('test_rag', 'How is data stored in milvus?')
print(json.dumps(kk, indent=4))
