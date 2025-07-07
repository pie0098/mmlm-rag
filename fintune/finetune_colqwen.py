from datasets import load_dataset
from PIL import Image
import os
from functools import partial

print("----------------Load Datasets------------------")

# load json file
json_data_path = "/home/linux/yyj/colpali/finetune/mmlm-rag/utils/img_query_pairs.json"
ds = load_dataset("json", data_files=json_data_path, split="train")

# load images
image_dir = "/home/linux/yyj/colpali/finetune/pdf2images_with_reference_per_city"
def load_image(example, image_dir):
    full_path = os.path.join(image_dir, example["image_path"])
    example["image"] = Image.open(full_path)
    return example

ds = ds.map(partial(load_image, image_dir=image_dir))

# split dataset
ds = ds.train_test_split(test_size=0.2, seed=42, shuffle=False)
train_ds = ds["train"]
test_ds = ds["test"]
print("----------------Load Model------------------")

import torch
from transformers import AutoConfig, Trainer, TrainingArguments, BitsAndBytesConfig, \
    ColQwen2ForRetrieval, ColQwen2Processor, EarlyStoppingCallback
from colpali_engine.loss import ColbertPairwiseCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from peft import LoraConfig, get_peft_model
torch.manual_seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_dir = "/home/linux/yyj/colpali/finetune/colqwen2-v1.0-hf"
model = ColQwen2ForRetrieval.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
).eval()


lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        init_lora_weights="gaussian"
    )
lora_config.inference_mode = False
model = get_peft_model(model, lora_config)
max_pixels = 1100*28*28
processor = ColQwen2Processor.from_pretrained(model_dir, max_pixels=max_pixels)


def collate_fn(examples):
    texts = []
    images = []

    for example in examples:

        texts.append(example["complex_query"])
        images.append(example["image"].convert("RGB"))

    batch_images = processor(images=images, return_tensors="pt").to(model.device)
    batch_queries = processor(text=texts, return_tensors="pt").to(model.device)
    return (batch_queries, batch_images)


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, num_items_in_batch=4, return_outputs=False):
        query_inputs, doc_inputs = inputs
        query_outputs = model(**query_inputs)
        doc_outputs = model(**doc_inputs)
        loss = self.loss_func(query_outputs.embeddings, doc_outputs.embeddings)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        query_inputs, doc_inputs = inputs # unpack from data collator
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            doc_outputs = model(**doc_inputs)

            loss = self.loss_func(query_outputs.embeddings, doc_outputs.embeddings)
            
            return loss, None, None if prediction_loss_only else loss

training_args = TrainingArguments(
    output_dir="./colpali_city_0702_colqwen2",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    logging_steps=10,
    eval_strategy="steps",    
    eval_steps=10,
    warmup_steps=20,
    learning_rate=2e-4,
    save_total_limit=1,
    report_to="tensorboard",
    dataloader_pin_memory=False,
    load_best_model_at_end=True,      
    metric_for_best_model="loss",      
    greater_is_better=False   
)


trainer = ContrastiveTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=training_args,
    loss_func=ColbertPairwiseCELoss(),
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.args.remove_unused_columns = False
print("----------------------------Start Training--------------------------")
trainer.train() 
print("----------------------------Save Model--------------------------")
trainer.save_model("/home/linux/yyj/colpali/finetune/wiky_city_zh_0702_lr2e4_unshuffle_colqwen")  
processor.save_pretrained("/home/linux/yyj/colpali/finetune/wiky_city_zh_0702_lr2e4_unshuffle_colqwen")
print("----------------------------Finish Training--------------------------")
