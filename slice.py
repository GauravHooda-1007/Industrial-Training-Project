import os
import nibabel as nib
import numpy as np
from PIL import Image

def save_slices_from_nifti(nifti_img_path, label, subject_id, output_root="C:/Users/gaura/Desktop/OAS/OAS_test", image_size=(160, 160)):
    print(f"🔍 Loading NIfTI: {nifti_img_path}")
    try:
        img = nib.load(nifti_img_path)
        data = np.squeeze(img.get_fdata())  # 🔧 Removes that extra (1,) dimension

        if data.ndim != 3:
            print(f"⚠️ Skipping {subject_id}, invalid shape: {data.shape}")
            return

        mid = data.shape[2] // 2
        half_range = 40  # choose how many before and after
        start = max(mid - half_range, 0)
        end = min(mid + half_range, data.shape[2])
        slices = [data[:, :, i] for i in range(start, end)]


        label_folder = os.path.join(output_root, label)
        os.makedirs(label_folder, exist_ok=True)

        for i, slice_2d in enumerate(slices):
            slice_norm = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255).astype(np.uint8)
            img_pil = Image.fromarray(slice_norm)
            img_pil = img_pil.resize(image_size)
            save_path = os.path.join(label_folder, f"{subject_id}_slice{i}.png")
            img_pil.save(save_path)

        print(f"✅ Saved slices for {subject_id} in {label}")

    except Exception as e:
        print(f"❌ Error with {subject_id}: {e}")


# MAIN DRIVER
base_dir = "C:/Users/gaura/Desktop/OAS/OAS_sorted"
output_dir = "C:/Users/gaura/Desktop/OAS/OAS_test"

labels = os.listdir(base_dir)

for label in labels:
    label_path = os.path.join(base_dir, label)
    if not os.path.isdir(label_path): continue

    for subject_folder in os.listdir(label_path):
        raw_path = os.path.join(label_path, subject_folder, "RAW")
        if not os.path.isdir(raw_path): continue

        for file in os.listdir(raw_path):
            if file.startswith("mpr-1") and file.endswith(".nifti.img"):  # 🔥 Correct extension now
                nifti_path = os.path.join(raw_path, file)
                save_slices_from_nifti(nifti_path, label, subject_folder, output_root=output_dir)
