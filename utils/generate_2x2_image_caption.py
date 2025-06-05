import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import argparse
import re


def save_progress(image_data, ImgCap_file):
    """
    Save the processing progress
    """
    try:
        with open(ImgCap_file, 'w', encoding='utf-8') as f:
            json.dump(image_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving progress: {str(e)}")
        return False

def construct_prompt(city_name: str, page_num: str = "") -> str :
    prompt = f"""你是城市信息分析专家，当前需要你分析的城市是{city_name}。请根据图中显示的内容进行描述，提取并总结该页的核心信息，请确保：
                1.只陈述图片中看到的文字、数字、图表等，不添加任何超出该图片内容的背景、原因或推断；
                2.保持描述的客观性和准确性，避免生成与图片无关的内容。
                回复时请用中文回复，语言简洁、逻辑清晰，全文控制在 60-80 字之间。如需引用图中文字，可用“”括起。
                """
    prompt1 = f"""你是城市信息分析专家，当前需要你分析的城市是{city_name}。请仔细查看我发送的 PDF 页面图像，提取并总结该页的核心信息。请按照以下结构输出：
                        1. **标题/主题**：一句话概括本页主要内容；  
                        2. **关键要点**：列出 3–5 条最重要的信息或结论；  
                        3. *图表说明**：简要概括图表、表格、图片等描述内容；  

                回复时请用中文回复，语言简洁、逻辑清晰，总长度不超过 500 字。如需引用图中文字，可用“”括起。  
                下面是第 {page_num} 页图像，请开始描述："""
    return prompt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images and generate query results')
    parser.add_argument('--model_dir', type=str, 
                      default="/home/linux/yyj/colpali/finetune/qwen2.5-vl-7b-instruct",
                      help='Model directory path')
    parser.add_argument('--image_root_dir', type=str, 
                      default="/home/linux/yyj/colpali/finetune/pdf2images_test",
                      help='Image directory path')
    parser.add_argument('--output_file', type=str, 
                      default='/home/linux/yyj/colpali/finetune/mmlm-rag/utils/split_2x2_image_cap_pairs.json',
                      help='Output JSON file path')
    parser.add_argument('--prompt_name', type=str,
                      default="self_defined",
                      help='Prompt template name')
    parser.add_argument('--max_pixels', type=int,
                      default=1280*28*28,
                      help='Maximum number of pixels')
    # Set max_new_tokens larger to prevent output results from not being in JSON format
    parser.add_argument('--max_new_tokens', type=int,
                      default=80,
                      help='Maximum number of tokens generated')
    
    args = parser.parse_args()
    
    # Verify if the paths exist
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}")
    if not os.path.exists(args.image_root_dir):
        raise ValueError(f"Image directory does not exist: {args.image_root_dir}")
    
    # Load the model and processor
    print(f"Loading model: {args.model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, max_pixels=args.max_pixels)
    
    # Read or create the JSON file
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
    else:
        # If the file does not exist, create a new data structure
        image_data = []
        for image_dir in os.listdir(args.image_root_dir):
            image_dir_path = os.path.join(args.image_root_dir, image_dir)

            for img_file in os.listdir(image_dir_path):
                image_path = os.path.join(image_dir_path, img_file)
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_data.append({
                        'image_path': image_path,
                        'generated': "False"
                    })
    
    # Iterate through all image paths
    for i, item in tqdm(enumerate(image_data), total=len(image_data), desc="Processing images"):
        # If it has been processed, skip
        if item.get('generated') == "True":
            continue
            
        image_path = item['image_path']
        if not os.path.exists(image_path):
            print(f"Warning: Image file does not exist: {image_path}")
            continue

        img_path_list = image_path.split('/')
        city_name = img_path_list[-2],   
        # page_num = re.search(r"\d+", img_path_list[-1]).group()
        prompt = construct_prompt(city_name=city_name)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: generate output
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"\nSaving image-captions: {image_path}")
        
        try:
            # Update the JSON data
            image_data[i]['caption'] = output_text[0]
            image_data[i]['generated'] = "True"
            
            print(f"Successfully updated the caption results for image: {image_path}")
            
            # Save progress after processing each image
            if save_progress(image_data, args.output_file):
                print(f"Saved current progress to {args.output_file}")
            else:
                print("Warning: Failed to save progress")
                
        except Exception as e:
            print(f"Error processing image: {image_path}: {str(e)}")
            image_data[i]['generated'] = "False"
            save_progress(image_data, args.output_file)

    print(f"\nAll results have been saved to {args.output_file}")

if __name__ == "__main__":
    main()