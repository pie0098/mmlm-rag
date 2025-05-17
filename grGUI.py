from transformers import ColPaliForRetrieval, ColPaliProcessor
import torch
from peft import PeftModel
import gradio as gr
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
# 加载 tokenizer 和基座模型
base_model_path = "/home/linux/yyj/colpali/finetune/colpali-v1.2-hf"
base_model = ColPaliForRetrieval.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
# 加载微调后的 adapter（LoRA 权重）
adapter_path = "/home/linux/yyj/colpali/finetune/wiki_city"
model = PeftModel.from_pretrained(base_model, adapter_path)
processor = ColPaliProcessor.from_pretrained(adapter_path)

def index(file, ds):
 
    images = []
    for f in file:
        images.extend(convert_from_path(f))
 
    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to("cuda") for k, v in batch_doc.items()}
            outputs = model(**batch_doc)
            embs = outputs.embeddings.to("cuda")
        ds.extend(list(torch.unbind(embs)))
    return f"Uploaded and converted {len(images)} pages", ds, images
 
 
def search(query: str, ds, images):
    qs = []
    with torch.no_grad():
        batch_query = processor.process_queries([query]).to("cuda")
        batch_query = {k: v.to("cuda") for k, v in batch_query.items()}
        outputs = model(**batch_query)
        embs = outputs.embeddings.to("cuda")
        qs.extend(list(torch.unbind(embs)))
 
    # run evaluation
    # retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    # scores = retriever_evaluator.evaluate(qs, ds)
    scores = processor.score_retrieval(qs, ds)
    best_page = int(scores.argmax(axis=1).item())
    return f"最相关的页面是 {best_page}", images[best_page]
 
 
COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
 
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
 
with gr.Blocks() as demo:
    gr.Markdown(
        "# ColPali: 基于视觉语言模型的高效文档检索系统 📚🔍"
    )
    gr.Markdown("## 1️⃣ 上传PDF文件")
    file = gr.File(file_types=[".pdf"], file_count="multiple")
 
    gr.Markdown("## 2️⃣ 索引PDF文件并上传")
    convert_button = gr.Button("🔄 转换并上传")
    message = gr.Textbox("文件尚未上传")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])
 
    # Define the actions for conversion
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])
 
    gr.Markdown("## 3️⃣ 搜索")
    query = gr.Textbox(placeholder="请输入您的查询内容", lines=5)
    search_button = gr.Button("🔍 搜索")
 
    gr.Markdown("## 4️⃣ ColPali检索结果")
    message2 = gr.Textbox("最相关的图片是...")
    output_img = gr.Image()

    gr.Markdown("## 5️⃣ 模型回答")
    output_text = gr.Textbox("模型回答...")
    # def get_answer(prompt: str, image: Image):
    #     response = gemini_flash.generate_content([prompt, image])
    #     return response.text

    def get_answer_qwen25vl_3b(prompt: str, image: Image):
        model_dir = "/home/linux/yyj/colpali/finetune/qwen2.5-vl-7b-instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        # min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen_processor = AutoProcessor.from_pretrained(model_dir, max_pixels=max_pixels)
        messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
        text = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
 
    # 用于Function to combine retrieval and LLM call 
    def search_with_llm(
        query,
        ds,
        images,
    ):
        # Step 1: Search the best image based on query
        search_message, best_image = search(query, ds, images)
 
        # Step 2: Generate an answer using LLM
        answer = get_answer_qwen25vl_3b(query, best_image)
 
        return search_message, best_image, answer
 
    # Action for search button
    search_button.click(
        search_with_llm,
        inputs=[query, embeds, imgs],
        outputs=[message2, output_img, output_text],
    )
 
if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True, share=True)