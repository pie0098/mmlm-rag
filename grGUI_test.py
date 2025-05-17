import torch
from peft import PeftModel
from transformers import (
    ColPaliForRetrieval,
    ColPaliProcessor,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
import gradio as gr
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path

# =======================================
# 1. 全局加载：只运行一次，并放在 CUDA
# =======================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (1) ColPali 检索模型 + Adapter
BASE_MODEL_PATH = "/home/linux/yyj/colpali/finetune/colpali-v1.2-hf"
ADAPTER_PATH    = "/home/linux/yyj/colpali/finetune/wiki_city"

colpali_base = ColPaliForRetrieval.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE}
)
colpali_model = PeftModel.from_pretrained(colpali_base, ADAPTER_PATH)
colpali_model.to(DEVICE)
colpali_processor = ColPaliProcessor.from_pretrained(ADAPTER_PATH)

# (2) Qwen2.5-VL 文图生成模型
QWEN_DIR   = "/home/linux/yyj/colpali/finetune/Qwen2.5-VL-3B-Instruct"
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_DIR,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE}
)
qwen_model.to(DEVICE)
qwen_processor = AutoProcessor.from_pretrained(QWEN_DIR, max_pixels=1280*28*28)

# =======================================
# 2. 索引函数：批量化 & 全部放 CUDA
# =======================================
def index(file_list, ds):
    images = []
    for f in file_list:
        images.extend(convert_from_path(f))
    loader = DataLoader(
        images, batch_size=8, shuffle=False,
        collate_fn=lambda imgs: colpali_processor.process_images(imgs)
    )
    for batch in tqdm(loader, desc="索引页面"):
        with torch.no_grad():
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = colpali_model(**batch)
            embs = out.embeddings  # b x seq_len x dim on DEVICE
            ds.extend(torch.unbind(embs))
    return f"已上传并索引 {len(images)} 页", ds, images

# =======================================
# 3. 检索函数：单 query 批量 & CUDA
# =======================================
def search(query: str, ds, images):
    with torch.no_grad():
        q_inputs = colpali_processor.process_queries([query])
        q_inputs = {k: v.to(DEVICE) for k, v in q_inputs.items()}
        out = colpali_model(**q_inputs)
        q_embs = out.embeddings  # 1 x seq_len x dim on DEVICE
        qs = list(torch.unbind(q_embs))
    scores = colpali_processor.score_retrieval(qs, ds)
    best_idx = int(scores.argmax())
    return f"最相关的页面是 {best_idx}", images[best_idx]

# =======================================
# 4. 多模态问答：全部 CUDA
# =======================================
FIXED_PROMPT = (
    "请详细分析这张图片的内容，包括但不限于：\n"
    "1. 图片的主要内容和主题\n"
    "2. 图片中的关键信息点\n"
    "3. 图片的布局和结构\n"
    "4. 图片中可能包含的重要数据或统计信息\n"
    "请以结构化的方式输出，确保信息清晰易读。"
)

def get_answer_qwen25vl(prompt: str, image: Image.Image):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=img_inputs,
        videos=vid_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=512)
    trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], output_ids)]
    return qwen_processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

# =======================================
# 5. 组合函数
# =======================================
def search_with_llm(query, ds, images):
    msg, best_img = search(query, ds, images)
    answer = get_answer_qwen25vl(query, best_img)
    return msg, best_img, answer

# =======================================
# 6. Gradio 界面配置
# =======================================
with gr.Blocks() as demo:
    gr.Markdown("# ColPali 文档 + 多模态问答系统 📚🔍")
    file    = gr.File(file_types=[".pdf"], file_count="multiple", label="上传 PDF")
    btn_idx = gr.Button("转换并索引")
    out_msg = gr.Textbox(label="状态")
    state_ds  = gr.State([])
    state_imgs= gr.State([])

    btn_idx.click(
        index, inputs=[file, state_ds], outputs=[out_msg, state_ds, state_imgs]
    )

    qry    = gr.Textbox(placeholder="输入查询", label="查询文本")
    btn_s  = gr.Button("搜索")
    out_msg2 = gr.Textbox(label="检索结果")
    out_img  = gr.Image(label="最佳页面")
    out_txt  = gr.Textbox(label="模型回答")

    btn_s.click(
        search_with_llm,
        inputs=[qry, state_ds, state_imgs],
        outputs=[out_msg2, out_img, out_txt]
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True, share=True)
