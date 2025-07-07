import os
import re

# 顶层目录
ROOT_DIR = "/home/linux/yyj/colpali/finetune/pdf2images_all_splits"

# 用来提取 “页号-分块号” 的正则：捕获两个数字组
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
            # 如果文件名格式不符合预期，可以选择跳过或打印警告
            print(f"⚠️ 跳过无法解析的文件: {os.path.join(sub, fn)}")
            continue

        page_str, chunk_str = m.groups()
        page_num  = int(page_str)
        chunk_num = int(chunk_str)

        new_name = f"page_{page_num:03d}_{chunk_num}.png"
        src = os.path.join(subdir, fn)
        dst = os.path.join(subdir, new_name)

        # 如果目标名已存在，可先删除或跳过
        if os.path.exists(dst):
            print(f"⚠️ 目标已存在，跳过: {dst}")
            continue

        os.rename(src, dst)
        print(f"{src}  →  {dst}")

print("✅ 重命名完成！")
