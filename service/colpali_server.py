from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from torch.utils.data import DataLoader
from peft import PeftModel
import torch
from typing import List
from PIL import Image
from tqdm import tqdm
import os

class ColQwen2Service:
    def __init__(self, base_model_path: str, adapter_path: str, max_pixels : int, device: str = "cuda:0") -> None:

        self.device = get_torch_device(device) 
        base_model = ColQwen2ForRetrieval.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.processor = ColQwen2Processor.from_pretrained(adapter_path, max_pixels=max_pixels, use_fast=True)  

    def process_queries(self, queries: List[str]) -> List[torch.Tensor]:
        qr_dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        qs: List[torch.Tensor] = []
        for batch_query in qr_dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                query_outputs = self.model(**batch_query)
                qr_embs = query_outputs.embeddings
            qs.extend(list(torch.unbind(qr_embs.to(self.device))))
        
        return qs

    def process_images(self, image_paths: str, token_pooling: bool = True) -> List[torch.Tensor]:
        images = [Image.open(image_paths)]

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        ds: List[torch.Tensor] = []
        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                image_outputs = self.model(**batch_doc)
                img_embs = image_outputs.embeddings
            ds.extend(list(torch.unbind(img_embs.to(self.device))))
        if token_pooling:
            pooler = HierarchicalTokenPooler()
            # Pool the embeddings, returun_dict default as False, only return pooled_embeddings
            outputs = pooler.pool_embeddings(ds, pool_factor=3)
        return outputs