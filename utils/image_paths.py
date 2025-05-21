import os
import json

def count_file_images(directory: str):
    """
    Count the total number of files in the specified directory and its subdirectories
    :param directory: Directory path to count files from, defaults to current directory
    :return: Total number of files
    """
    try:
        total_files = 0
        # Traverse directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            # Count files in current directory
            total_files += len(files)
            
        print(f"Find {total_files} files in directory '{directory}' and its subdirectories")
        return total_files
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return -1

def get_image_paths(new_folders):
    """
    get specific directory all image file path list
    support image format: .jpg, .jpeg, .png 
    """
    image_extensions = ('.jpg', '.jpeg', '.png')
    upload_image_paths = []

    for folder in new_folders:
        for images_file in os.listdir(folder):
            if images_file.lower().endswith(image_extensions):
                full_path = os.path.join(folder, images_file)
                if full_path not in upload_image_paths:
                    upload_image_paths.append(full_path)
    
    return upload_image_paths

def get_new_folders(directory, manifest_file):

    try:
        with open(manifest_file, "r") as f:
            existing = json.load(f)

    except FileNotFoundError:
        print(f"Note: Manifest file {manifest_file} not found, will create a new one")
        existing = {"files": [], "file_image_counts": 0}

    previous_file_image_num = existing["file_image_counts"]

    current = {"files": [], "file_image_counts": 0}
    current["files"] = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    # 1 refer to the manifest_file
    current["file_image_counts"] = count_file_images(directory)
    
    with open(manifest_file, "w") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    
    cur = current["files"]  
    old = existing["files"] 

    new_folders     = list(set(cur) - set(old))
    previous_folders = list(set(cur) & set(old))

    return new_folders, previous_folders, previous_file_image_num
