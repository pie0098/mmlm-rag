from colpali_server import ColQwen2Service
from milvus_retriever import MilvusColbertRetriever
from pymilvus import MilvusClient
import time

# Get model service
print("---------------------Get model service---------------------\n")

base_model_path = "/home/linux/yyj/colpali/finetune/colqwen2-v1.0-hf"
adapter_path = "/home/linux/yyj/colpali/finetune/wiky_city_zh_0702_lr2e4_colqwen"
max_pixels = 1100*28*28
model_service = ColQwen2Service(base_model_path, adapter_path, max_pixels)

# Get milvus service
print("---------------------Get milvus service---------------------\n")
client = MilvusClient(uri="http://localhost:19530")
# before inserting data, collection must be loaded in attu
retriever = MilvusColbertRetriever(collection_name="colqwen", milvus_client=client)

print("--------------------Test service---------------------------\n")

queries = [
    "德里12月降水量怎么样",
    "爱丁堡5月气温怎么样",
]
qs = model_service.process_queries(queries)
for i in range(len(qs)):
    # query: 22*128
    start_time = time.time()
    query = qs[i].float().cpu().numpy()
    result = retriever.search(query, topk=1)
    end_time = time.time()
    print(f"query is {queries[i]} \n")
    print(f"result is {result[0][1]} \n")
    print(f"本次检索耗时: {end_time - start_time:.4f} 秒\n")