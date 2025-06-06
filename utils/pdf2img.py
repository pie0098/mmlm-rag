from pdf2image import convert_from_path
import os
from PIL import Image
import argparse

import cv2
import os
import json
from typing import Optional

def split_2x2_image_and_save_json(image_dir, output_dir, saved_json: Optional[bool] = False):
    
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for dir in os.listdir(image_dir):

        old_dir = os.path.join(image_dir, dir)
        new_dir = os.path.join(output_dir, dir)

        os.makedirs(new_dir, exist_ok=True)
        if len(os.listdir(new_dir)):
                continue

        for file in os.listdir(old_dir):

            old_file_path = os.path.join(old_dir, file)
            # 读取图片
            img = cv2.imread(old_file_path)
            if img is None:
                raise ValueError(f"无法读取图片: {old_file_path}")
            
            # 获取图片尺寸
            height, width = img.shape[:2]
            half_w, half_h = width // 2, height // 2
            # 定义四个子区域
            quadrants = [
                img[0:half_h,       0:half_w],   # 左上
                img[0:half_h,       half_w:width], # 右上
                img[half_h:height,  0:half_w],   # 左下
                img[half_h:height,  half_w:width]  # 右下
            ]
            
            # 保存结果信息
            
            # 保存每个象限
            for i, q in enumerate(quadrants, start=1):
                # 生成保存路径
                base_name = os.path.splitext(os.path.basename(file))[0]
                save_path = os.path.join(new_dir, f"{base_name}_q{i}.png")
                # 保存图片
                cv2.imwrite(save_path, q)
                # 记录信息
                results.append({
                    "image_path": save_path,
                    "caption" : "",
                    "generated" : ""
                })
    if saved_json:
        try:
            file_name = "split_2x2_image_cap_pairs.json"
            json_path = os.path.join(output_dir, file_name)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"save results in file {json_path}")
        except Exception as e:
            print(f"save results in file with error: {e}")
# image_dir = "/home/linux/yyj/colpali/finetune/pdf2images"
# output_dir = "/home/linux/yyj/colpali/finetune/pdfTo_2x2_image"
# split_2x2_image_and_save_json(image_dir, output_dir, saved_json=True)

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300, fmt: str = 'PNG'):
    """
    Convert a PDF file to images    

    Args:
    - pdf_path: The path of the PDF file
    - output_dir: The output directory
    - dpi: The image resolution
    - fmt: The output image format (PNG/JPEG)
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing PDF file: {pdf_path}")
    try:
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt=fmt.lower(),
            poppler_path=r"D:\\poputils\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin"  # Windows needs to set
        )
        
        # Get the PDF file name (without extension) as the subfolder name
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        
        # Create a unique output directory for each PDF
        if not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir)
        
        # Save images
        for i, image in enumerate(images):
            # Build the output file name
            output_file = os.path.join(
                pdf_output_dir, 
                f"page_{i+1:03d}.{fmt.lower()}"
            )
            
            # Save images
            image.save(output_file, fmt)
            print(f"Saved page {i+1}: {output_file}")
            
        print(f"\nConversion completed! Processed {len(images)} pages")
        print(f"Output directory: {pdf_output_dir}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")     

def process_pdf_folder(input_dir: str, output_dir: str, dpi: int = 300, fmt: str = 'PNG'):
    """
    Process all PDF files in a folder

    Args:
    - input_dir: The input folder path
    - output_dir: The output directory
    - dpi: The image resolution
    - fmt: The output image format (PNG/JPEG)
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"The input directory does not exist: {input_dir}")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        pdf_to_images(pdf_path, output_dir, dpi, fmt)

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Convert PDF files in a folder to images')
    parser.add_argument('input_dir', help='The path of the PDF folder') 
    parser.add_argument(
        '--output_dir', 
        '-o', 
        help='The output directory',
        default='output_images'
    )
    parser.add_argument(
        '--dpi', 
        '-d', 
        type=int, 
        help='The image resolution',
        default=200
    )
    parser.add_argument(
        '--format', 
        '-f', 
        help='The output image format (PNG/JPEG)',
        default='PNG'
    )
    
    args = parser.parse_args()
    
    # Execute conversion
    process_pdf_folder(
        args.input_dir,
        args.output_dir,
        args.dpi,
        args.format
    )

if __name__ == "__main__":
    main()
