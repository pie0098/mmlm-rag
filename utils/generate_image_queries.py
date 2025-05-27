import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from retrieval_prompts_cn import get_retrieval_prompt
from pydantic import BaseModel
import os
from tqdm import tqdm
import argparse

class QueryModel(BaseModel):
    simple_query: str
    simple_explanation: str
    complex_query: str
    complex_explanation: str
    text_visual_combination_query: str
    text_visual_combination_explanation: str

def parse_query_json(json_str: str) -> QueryModel:
    """
    Decode the query JSON string
    """
    # Remove possible markdown code block markers
    json_str = json_str.strip('`').strip('json').strip()
    
    # Parse the JSON
    data = json.loads(json_str)
    
    # Convert to Pydantic model
    query_model = QueryModel(**data)
    return query_model

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images and generate query results')
    parser.add_argument('--model_dir', type=str, 
                      default="/home/linux/yyj/colpali/finetune/qwen2.5-vl-7b-instruct",
                      help='Model directory path')
    parser.add_argument('--image_dir', type=str, 
                      default="/home/linux/yyj/colpali/finetune/pdf2images",
                      help='Image directory path')
    parser.add_argument('--output_file', type=str, 
                      default='img_cap_pairs.json',
                      help='Output JSON file path')
    parser.add_argument('--prompt_name', type=str,
                      default="self_defined",
                      help='Prompt template name')
    parser.add_argument('--max_pixels', type=int,
                      default=1280*28*28,
                      help='Maximum number of pixels')
    # Set max_new_tokens larger to prevent output results from not being in JSON format
    parser.add_argument('--max_new_tokens', type=int,
                      default=512,
                      help='Maximum number of tokens generated')
    
    args = parser.parse_args()
    
    # Verify if the paths exist
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}")
    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    
    # Load the model and processor
    print(f"Loading model: {args.model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, max_pixels=args.max_pixels)
    
    # Get the prompt
    prompt, pydantic_model = get_retrieval_prompt(args.prompt_name)
    
    # Read or create the JSON file
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
    else:
        # If the file does not exist, create a new data structure
        image_data = []
        for img_file in os.listdir(args.image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_data.append({
                    'image_path': img_file,
                    'generated': "False"
                })
    
    # Iterate through all image paths
    for i, item in tqdm(enumerate(image_data), total=len(image_data), desc="Processing images"):
        # If it has been processed, skip
        if item.get('generated') == "True":
            continue
            
        image_path = os.path.join(args.image_dir, item['image_path'])
        if not os.path.exists(image_path):
            print(f"Warning: Image file does not exist: {image_path}")
            continue
            
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
        print(f"\nProcessing image: {image_path}")
        
        try:
            # Parse the output results
            query_model = parse_query_json(output_text[0])
            
            # Update the JSON data
            image_data[i]['simple_query'] = query_model.simple_query
            image_data[i]['simple_explanation'] = query_model.simple_explanation
            image_data[i]['complex_query'] = query_model.complex_query
            image_data[i]['complex_explanation'] = query_model.complex_explanation
            image_data[i]['text_visual_combination_query'] = query_model.text_visual_combination_query
            image_data[i]['text_visual_combination_explanation'] = query_model.text_visual_combination_explanation
            image_data[i]['generated'] = "True"
            
            print(f"Successfully updated the query results for image: {image_path}")
            
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
