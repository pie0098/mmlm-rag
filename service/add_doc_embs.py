from colpali_server import ColQwen2Service
from milvus_retriever import MilvusColbertRetriever
from pymilvus import MilvusClient
import os
import time
from tqdm import tqdm

# Get milvus service
print("---------------------Get milvus service---------------------\n")
client = MilvusClient(uri="http://localhost:19530")
# before inserting data, collection must be loaded in attu
retriever = MilvusColbertRetriever(collection_name="colqwen", milvus_client=client)
retriever.create_collection()
retriever.create_index()

# Get model service
print("---------------------Get model service---------------------\n")

base_model_path = "/home/linux/yyj/colpali/finetune/colqwen2-v1.0-hf"
adapter_path = "/home/linux/yyj/colpali/finetune/wiky_city_zh_0702_lr2e4_colqwen"
max_pixels = 1100*28*28
model_service = ColQwen2Service(base_model_path, adapter_path, max_pixels)

print("--------------------Insert doc embeddings to milvus---------------------------\n")

pages_dir = "/home/linux/yyj/colpali/finetune/pdf2images_with_reference_per_city"
# pages_dir = "/home/linux/yyj/colpali/finetune/mmlm-rag/test_pages"
# filepaths = [os.path.join(pages_dir, name) for name in os.listdir(pages_dir)]
filepaths = []
for dir in os.listdir(pages_dir):
    dirpath = os.path.join(pages_dir, dir)
    for name in os.listdir(dirpath):
        filepaths.append(os.path.join(dirpath, name))
        
for i in tqdm(range(len(filepaths))):
    ds = model_service.process_images(filepaths[i])
    data = {
        "colbert_vecs": ds[0].float().cpu().numpy(),
        "doc_id": i,
        "filepath": filepaths[i],
    }
    retriever.insert(data)

