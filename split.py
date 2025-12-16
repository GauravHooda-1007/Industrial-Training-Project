import os
import shutil
import random
from tqdm import tqdm

source_dir = "C:/Users/gaura/Desktop/OAS/OAS_png_dataset"
output_dir = "C:/Users/gaura/Desktop/OAS/OAS_split_dataset"
val_ratio = 0.2  # 20% validation

random.seed(42)

for label in os.listdir(source_dir):
    class_dir = os.path.join(source_dir, label)
    if not os.path.isdir(class_dir):
        continue

    images = os.listdir(class_dir)
    random.shuffle(images)

    split_idx = int(len(images) * (1 - val_ratio))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for mode, img_list in zip(["train", "val"], [train_imgs, val_imgs]):
        dest_dir = os.path.join(output_dir, mode, label)
        os.makedirs(dest_dir, exist_ok=True)

        for img_name in tqdm(img_list, desc=f"{mode.upper()} {label}"):
            src_path = os.path.join(class_dir, img_name)
            dst_path = os.path.join(dest_dir, img_name)
            shutil.copy2(src_path, dst_path)
