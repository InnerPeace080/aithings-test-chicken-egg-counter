import os
import random
import shutil

# Paths
base_dir = "data"
images_dir = os.path.join(base_dir, "labeled_data/images")
labels_dir = os.path.join(base_dir, "labeled_data/labels")

train_img_dir = os.path.join(base_dir, "train/images")
train_lbl_dir = os.path.join(base_dir, "train/labels")
test_img_dir = os.path.join(base_dir, "test/images")
test_lbl_dir = os.path.join(base_dir, "test/labels")

# remove existing train/test directories if they exist
for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)

# Create output directories
for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)


# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# Split ratio
split_ratio = 0.8
split_idx = int(len(image_files) * split_ratio)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]


def copy_files(file_list, img_dst, lbl_dst):
    for img_file in file_list:
        # Copy image
        shutil.copy2(os.path.join(images_dir, img_file), os.path.join(img_dst, img_file))
        # Copy label (replace image extension with .txt)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_src = os.path.join(labels_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(lbl_dst, label_file))


copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(test_files, test_img_dir, test_lbl_dir)
# copy meta file
classes_src = os.path.join(base_dir, "labeled_data", "classes.txt")
notes_src = os.path.join(base_dir, "labeled_data", "notes.json")
shutil.copy2(classes_src, os.path.join(base_dir, "train", "classes.txt"))
shutil.copy2(classes_src, os.path.join(base_dir, "test", "classes.txt"))
shutil.copy2(notes_src, os.path.join(base_dir, "train", "notes.json"))
shutil.copy2(notes_src, os.path.join(base_dir, "test", "notes.json"))

print(f"Train set: {len(train_files)} images")
print(f"Test set: {len(test_files)} images")
