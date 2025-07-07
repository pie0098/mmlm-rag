import os
import re

# Top-level directory
ROOT_DIR = "/home/linux/yyj/colpali/finetune/pdf2images_all_splits"

# Regex to extract "page number-chunk number": captures two groups of digits
PAT = re.compile(r"(\d+)[-_](\d+)")

for sub in os.listdir(ROOT_DIR):
    subdir = os.path.join(ROOT_DIR, sub)
    if not os.path.isdir(subdir):
        continue

    for fn in os.listdir(subdir):
        if not fn.lower().endswith(".png"):
            continue
        m = PAT.search(fn)
        if not m:
            # If the filename format does not match the expectation, you can choose to skip or print a warning
            print(f"⚠️ Skipping unparseable file: {os.path.join(sub, fn)}")
            continue

        page_str, chunk_str = m.groups()
        page_num  = int(page_str)
        chunk_num = int(chunk_str)

        new_name = f"page_{page_num:03d}_{chunk_num}.png"
        src = os.path.join(subdir, fn)
        dst = os.path.join(subdir, new_name)

        # If the target name already exists, you can delete or skip it first
        if os.path.exists(dst):
            print(f"⚠️ Target already exists, skipping: {dst}")
            continue

        os.rename(src, dst)
        print(f"{src}  →  {dst}")

print("✅ Renaming completed!")
