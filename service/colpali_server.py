from colpali_engine.models import ColPali
from transformers import ColPaliForRetrieval, ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
from typing import cast

class ColPaliService:
    def __init__(self, model_dir: str, device: str = "cuda:0") -> None:

        self.device = get_torch_device(device) 
        self.model = ColPaliForRetrieval.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_dir))

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

    def process_images(self, image_paths: List[str]) -> List[torch.Tensor]:
        images = [Image.open(image_path) for image_path in image_paths]

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                image_outputs = self.model(**batch_doc)
                img_embs = image_outputs.embeddings
            ds.extend(list(torch.unbind(img_embs.to(self.device))))
        return ds