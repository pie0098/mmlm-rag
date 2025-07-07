import sys
import os

# __file__ is the absolute path of app.py, go up two levels to the mmlm-rag root directory
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root)

import gradio as gr
from service.colpali_server import ColQwen2Service
from service.milvus_retriever import MilvusColbertRetriever
from service.qwenvl_service import QwenVlService
from pymilvus import MilvusClient
from PIL import Image

# Initialize retrieval and generation services
base_model_path = "/home/linux/yyj/colpali/finetune/colqwen2-v1.0-hf"
adapter_path = "/home/linux/yyj/colpali/finetune/wiky_city_zh_0702_lr2e4_colqwen"
max_pixels = 1100 * 28 * 28
model_service = ColQwen2Service(base_model_path, adapter_path, max_pixels)
client = MilvusClient(uri="http://localhost:19530")
retriever = MilvusColbertRetriever(collection_name="colqwen", milvus_client=client)
generator_path = "/home/linux/yyj/colpali/finetune/Qwen2.5-VL-3B-Instruct"
generator = QwenVlService(generator_path, max_pixels=max_pixels)


def bot_response(user_message, chat_history):
    # Retrieve the best page, vectorize and search
    qs = model_service.process_queries([user_message])
    query_vec = qs[0].float().cpu().numpy()
    result = retriever.search(query_vec, topk=1)
    _, page_path = result[0][0], result[0][1]

    # Read the retrieved image
    retrieved_image = Image.open(page_path)

    # Call the generation model
    gen_text = generator.generate(user_message, retrieved_image)

    # Update chat history
    chat_history = chat_history or []
    chat_history.append((user_message, gen_text))

    return chat_history, chat_history, retrieved_image

# Build Gradio interface
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
        "# 🚀 ColQwen: Efficient VLM-based Document Retrieval System 📚🔍"
    )
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat Area",
                avatar_images=("/home/linux/yyj/colpali/finetune/mmlm-rag/app/images.jpg", "/home/linux/yyj/colpali/finetune/mmlm-rag/app/Usagi_main.webp"),
                show_copy_button=True,
                height=500
            )
        with gr.Column(scale=1):
            img_output = gr.Image(
                label="Retrieved Image",
                height=500
            )
    state = gr.State([])
    # Input area in a separate row, aligned with main content width
    with gr.Row(elem_id="input-row"):
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Enter your question...",
            lines=1,
            scale=8
        )
        send_btn = gr.Button("Send", scale=2)

    # Support sending with Enter key
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