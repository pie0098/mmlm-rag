from pdf2image import convert_from_path
import os
from PIL import Image
import argparse

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