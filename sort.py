import os
import shutil
import pandas as pd

# --- CONFIG ---
excel_path = r"C:\Users\gaura\Desktop\OAS\oas.xlsx"
source_dir = r"C:\Users\gaura\Desktop\OAS\OAS2"
output_dir = r"C:\Users\gaura\Desktop\OAS\OAS_sorted"

# --- Map CDR to Class Label ---
def cdr_to_label(cdr):
    if cdr == 0: return "NC"
    elif cdr == 0.5: return "EMCI"
    elif cdr == 1.0: return "LMCI"
    elif cdr >= 2.0: return "AD"
    return None

# --- Load and Filter Excel ---
df = pd.read_excel(excel_path)
df = df.dropna(subset=["CDR"])
df["Label"] = df["CDR"].map(cdr_to_label)

# --- Process ---
moved = 0
skipped = 0

for _, row in df.iterrows():
    mri_id = row["MRI ID"]           # e.g., OAS2_0001_MR1
    label = row["Label"]             # e.g., NC, EMCI, ...

    if not label:
        print(f"⚠️ Skipping {mri_id} due to unknown CDR label")
        skipped += 1
        continue

    src_path = os.path.join(source_dir, mri_id)
    dest_path = os.path.join(output_dir, label, mri_id)

    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)
        print(f"✅ Moved: {mri_id} → {label}")
        moved += 1
    else:
        print(f"❌ Folder not found: {src_path}")
        skipped += 1

print(f"\n🎉 Done! Moved: {moved}, Skipped: {skipped}")
