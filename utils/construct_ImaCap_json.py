import os
import json
import argparse

def get_all_file_paths(folder):
    file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), folder)
            if "\\" in rel_path:
                rel_path = rel_path.replace("\\", "/") 
            file_paths.append(rel_path)
    return file_paths

def build_json_list(file_paths):
    json_list = []
    for path in file_paths:
        json_list.append({
            "image_url": path,
            "simple_query": "",
            "simple_explanation": "",
            "complex_query": "",
            "complex_explanation": "",
            "text_visual_combination_query": "",
            "text_visual_combination_explanation": "",
            "generated": ""
        })
    return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save the image path in a JSON file')
    parser.add_argument('folder_path', help='The path of the folder containing the images')
    parser.add_argument('--output', '-o', default='img_cap_pairs.json', help='The path of the output JSON file (default: img_cap_pairs.json)')
    
    args = parser.parse_args()
    
    all_files = get_all_file_paths(args.folder_path)
    result = build_json_list(all_files)

    # Save the result as a JSON file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"The result has been saved to {args.output}") 