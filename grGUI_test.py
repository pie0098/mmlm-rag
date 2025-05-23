import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gradio as gr
import json
from typing import List
import multiprocessing
from pdf2image import convert_from_path

MAX_WORKERS = min(32, multiprocessing.cpu_count())
# =======================================
# 6. Gradio Interface Configuration
# =======================================

def save_uploaded_file(
        files, save_file_dir, save_image_dir, state_f, dpi=300, fmt="png", 
        max_workers=MAX_WORKERS
):
    
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_file_dir, exist_ok=True)
    
    dst_paths = []
    for file in files:
        src = file.name
        dst = os.path.join(save_file_dir, os.path.basename(src))
        dst_paths.append(dst)
        shutil.copy2(src, dst)
        state_f.append(f"✔ Saved PDF: {dst}")

        yield "\n".join(state_f), state_f

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_from_path, path, dpi=dpi, fmt=fmt.lower(), thread_count=3) : path
            for path in dst_paths
        }
        
        for future in as_completed(futures):
            pdf_path = futures[future]
            name = os.path.splitext(os.path.basename(pdf_path))[0]
            image_out_dir = os.path.join(save_image_dir, name)
            os.makedirs(image_out_dir, exist_ok=True)

            try:
                images = future.result()
                state_f.append(f"✔ Start to save images from {pdf_path}")
                yield "\n".join(state_f), state_f
                print(f"✔ Start to save images from {pdf_path}")

            except Exception as e:
                state_f.append(f"✖ Error converting {pdf_path}: {e}")
                yield "\n".join(state_f), state_f
                print(f"✖ Error converting {pdf_path}: {e}")

                continue
            n_pages = len(images)
            for idx, img in enumerate(images, start=1):
                # frac = idx / n_pages
                # progress(frac, desc=f"Saving page {idx}/{n_pages}")
                img_out_file = os.path.join(image_out_dir, f"page_{idx:03d}.{fmt.lower()}")
                img.save(img_out_file, fmt)
                state_f.append(f"✔ Converted {n_pages} pages for {name}, Saved page {idx:03d}: {image_out_dir}")
                yield "\n".join(state_f), state_f
                print(f"✔ Converted {n_pages} pages for {name}, Saved page {idx:03d}: {image_out_dir}")
            
            state_f.append(f"✔ Converted {n_pages} pages for {name}, Saved page {idx:03d}: {image_out_dir}")
            yield "\n".join(state_f), state_f
            print(f"✔ Sucessfully convert {name} to images at {image_out_dir} with {n_pages} pages !")
    
    state_f.append("🎉 All done !")
    yield "\n".join(state_f), state_f
    print("🎉 All done !")



with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("# ColPali Document + Multimodal QA System 📚🔍")
    
    with gr.Tabs():
 
        with gr.TabItem("Upload"):
            gr.Markdown("## File Upload Interface")
            
            with gr.Column():
                with gr.Row(equal_height=True):
                    save_file_dir = gr.Textbox(
                        label="Saved File Directory",
                        placeholder="Enter directory path to save files",
                        value="/home/linux/yyj/colpali/mmlm-rag/file_uploads"
                    )                    
                    save_image_dir = gr.Textbox(
                        label="Saved Image Directory",
                        placeholder="Enter directory path to save files",
                        value="/home/linux/yyj/colpali/mmlm-rag/test_pages"
                    )
                with gr.Row():
                    btn_file = gr.Button("Convert File into image")
                    btn_dir = gr.Button("Upload Image Embeddings to Milvus")
            
            files = gr.Files(
                file_types=[".pdf"],
                label="Upload PDF File",
                type="filepath"
            )
            with gr.Row():
                file_status = gr.Textbox(label="File Status", lines=10, max_lines=10, show_copy_button=True)
                image_status = gr.Textbox(label="Image Status", lines=10, max_lines=10, show_copy_button=True)
            
            state_f = gr.State([])
            state_imgs = gr.State([])
        
            btn_file.click(
                save_uploaded_file,
                inputs=[files, save_file_dir, save_image_dir, state_f],
                outputs=[file_status, state_f]
            )


        with gr.TabItem("Train"):
            gr.Markdown("## Training Interface")
            train_input = gr.Textbox(label="Training Data")
            train_btn = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Status")
            
        with gr.TabItem("Chat"):
            gr.Markdown("## Chat Interface")
            qry = gr.Textbox(placeholder="Enter your query", label="Query Text")
            btn_s = gr.Button("Search")
            out_msg2 = gr.Textbox(label="Search Results")
            out_img = gr.Image(label="Best Matching Page")
            out_txt = gr.Textbox(label="Model Response")
            

if __name__ == "__main__":
    demo.queue().launch()