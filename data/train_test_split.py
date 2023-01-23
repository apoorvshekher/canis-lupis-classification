import os
import shutil
import random

def split_dataset(input_folder, output_folder, split_ratio=(0.8, 0.1, 0.1)):
    if sum(split_ratio) != 1.0:
        raise ValueError("Split ratios should add up to 1.0")

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_folder, split)
        os.makedirs(split_path, exist_ok=True)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(images)

            total_images = len(images)
            train_size = int(split_ratio[0] * total_images)
            val_size = int(split_ratio[1] * total_images)
            test_size = total_images - train_size - val_size

            train_set = images[:train_size]
            val_set = images[train_size:train_size + val_size]
            test_set = images[train_size + val_size:]

            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(output_folder, split, class_folder), exist_ok=True)

            for image in train_set:
                source_path = os.path.join(class_path, image)
                dest_path = os.path.join(output_folder, 'train', class_folder, image)
                shutil.copy(source_path, dest_path)

            for image in val_set:
                source_path = os.path.join(class_path, image)
                dest_path = os.path.join(output_folder, 'val', class_folder, image)
                shutil.copy(source_path, dest_path)

            for image in test_set:
                source_path = os.path.join(class_path, image)
                dest_path = os.path.join(output_folder, 'test', class_folder, image)
                shutil.copy(source_path, dest_path)

if __name__ == "__main__":
    input_dataset_folder = 'Images'
    output_dataset_folder = 'images'
    split_dataset(input_dataset_folder, output_dataset_folder, split_ratio=(0.8, 0.1, 0.1))
