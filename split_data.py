import os
import shutil
import random

input_dir = r"tmp\data_wavelet"
output_dir = r"tmp\data_wavelet_out"
train_ratio = 0.8
seed = 42
random.seed(seed)

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]

def copy_files(files, dest):
    safe_mkdir(dest)
    for src in files:
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))

for cls in ["real", "fake"]:
    src_folder = os.path.join(input_dir, cls)
    if not os.path.isdir(src_folder):
        print(f"Không tìm thấy thư mục {src_folder}")
        continue

    images = list_images(src_folder)
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    train_files = images[:n_train]
    val_files = images[n_train:]

    train_dst = os.path.join(output_dir, "train", cls)
    val_dst = os.path.join(output_dir, "val", cls)

    copy_files(train_files, train_dst)
    copy_files(val_files, val_dst)

    print(f"{cls}: train={len(train_files)}, val={len(val_files)}, total={len(images)}")

print("\nHoàn tất! Cấu trúc output:")
print(f"{output_dir}/train/real/")
print(f"{output_dir}/train/fake/")
print(f"{output_dir}/val/real/")
print(f"{output_dir}/val/fake/")