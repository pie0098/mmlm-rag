import os
from pathlib import Path
import sys

def count_files(directory):
    """
    Recursively count all files in the specified directory
    
    Args:
        directory (str): The path of the directory to count
        
    Returns:
        tuple: (Total number of files, Total number of directories)
    """
    total_files = 0
    total_dirs = 0
    
    # Use Path object for directory traversal
    for path in Path(directory).rglob('*'):
        if path.is_file():
            total_files += 1
        elif path.is_dir():
            total_dirs += 1
            
    return total_files, total_dirs

def main():
    # Support command line arguments to specify directory
    if len(sys.argv) > 1:
        current_dir = sys.argv[1]
    else:
        current_dir = os.getcwd()
    
    # Count files
    files, directories = count_files(current_dir)
    
    # Print results
    print(f"\nResults:")
    print(f"Current directory: {current_dir}")
    print(f"Total number of files: {files}")
    print(f"Total number of directories: {directories}")
    
    # Count files by type
    print("\nCount files by type:")
    file_types = {}
    for path in Path(current_dir).rglob('*'):
        if path.is_file():
            ext = path.suffix.lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['no extension'] = file_types.get('no extension', 0) + 1
    
    for ext, count in sorted(file_types.items()):
        print(f"{ext}: {count} files")

if __name__ == "__main__":
    main() 