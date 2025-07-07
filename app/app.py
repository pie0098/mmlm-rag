import sys
import os

# __file__ 是 app.py 的绝对路径，往上两级到 mmlm‑rag 根目录
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root)

import gradio as gr
from service.colpali_server import ColQwen2Service
from service.milvus_retriever import MilvusColbertRetriever
from service.qwenvl_service import QwenVlService
from pymilvus import MilvusClient
from PIL import Image

# 初始化检索和生成服务
base_model_path = "/home/linux/yyj/colpali/finetune/colqwen2-v1.0-hf"
adapter_path = "/home/linux/yyj/colpali/finetune/wiky_city_zh_0702_lr2e4_colqwen"
max_pixels = 1100 * 28 * 28
model_service = ColQwen2Service(base_model_path, adapter_path, max_pixels)
client = MilvusClient(uri="http://localhost:19530")
retriever = MilvusColbertRetriever(collection_name="colqwen", milvus_client=client)
generator_path = "/home/linux/yyj/colpali/finetune/Qwen2.5-VL-3B-Instruct"
generator = QwenVlService(generator_path, max_pixels=max_pixels)


def bot_response(user_message, chat_history):
    # 检索最佳页面向量化并搜索
    qs = model_service.process_queries([user_message])
    query_vec = qs[0].float().cpu().numpy()
    result = retriever.search(query_vec, topk=1)
    _, page_path = result[0][0], result[0][1]

    # 读取检索到的图像
    retrieved_image = Image.open(page_path)

    # 调用生成模型
    gen_text = generator.generate(user_message, retrieved_image)

    # 更新对话历史
    chat_history = chat_history or []
    chat_history.append((user_message, gen_text))

    return chat_history, chat_history, retrieved_image

# 搭建 Gradio 界面
demo = gr.Blocks(css="""
#input-row .gr-textbox, #input-row .gr-button {
    height: 48px !important;
    min-height: 48px !important;
    box-sizing: border-box;
}
#input-row {
    align-items: center !important;
    display: flex !important;
}
""")
with demo:
    gr.Markdown(
        "# 🚀 ColQwen: 基于VLM的高效文档检索系统 📚🔍"
    )
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="对话区",
                avatar_images=("/home/linux/yyj/colpali/finetune/mmlm-rag/app/images.jpg", "/home/linux/yyj/colpali/finetune/mmlm-rag/app/Usagi_main.webp"),
                show_copy_button=True,
                height=500
            )
        with gr.Column(scale=1):
            img_output = gr.Image(
                label="检索到的图片",
                height=500
            )
    state = gr.State([])
    # 输入区单独一行，和主内容宽度对齐
    with gr.Row(elem_id="input-row"):
        user_input = gr.Textbox(
            show_label=False,
            placeholder="输入你的问题...",
            lines=1,
            scale=8
        )
        send_btn = gr.Button("发送", scale=2)

    # 支持回车发送
    user_input.submit(
        fn=bot_response,
        inputs=[user_input, state],
        outputs=[chatbot, state, img_output]
    )
    send_btn.click(
        fn=bot_response,
        inputs=[user_input, state],
        outputs=[chatbot, state, img_output]
    )
if __name__ == "__main__":
    demo.launch()