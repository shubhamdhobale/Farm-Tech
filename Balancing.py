import os
import cv2
import albumentations as A
import random
import shutil

# Define Dataset Paths
DATASET_DIR = '7 Crop Recommendation from Soil Image/Dataset'
OUTPUT_DIR = '7 Crop Recommendation from Soil Image/Balanced Dataset'
TARGET_COUNT = 400  # Desired number of images per class

# Define augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
])

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to copy images
def copy_images(src_dir, dst_dir):
    for img in os.listdir(src_dir):
        if img.endswith(('.jpg', '.png', '.jpeg')):
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(dst_dir, img)
            shutil.copy(src_path, dst_path)

# Function to augment images
def augment_images(class_dir, output_class_dir, target_count):
    images = [img for img in os.listdir(output_class_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)
    image_index = 0
    
    while current_count < target_count:
        img_path = os.path.join(output_class_dir, images[image_index % len(images)])
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Failed to read {img_path}, skipping.")
            image_index += 1
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = augmentations(image=image)
        aug_image = augmented['image']
        
        output_path = os.path.join(output_class_dir, f"aug_{current_count}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        
        current_count += 1
        image_index += 1

# Process each class
for soil_class in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, soil_class)
    output_class_dir = os.path.join(OUTPUT_DIR, soil_class)
    os.makedirs(output_class_dir, exist_ok=True)
    
    # Step 1: Copy all existing images to the output directory
    copy_images(class_dir, output_class_dir)
    
    # Step 2: Count images after copying
    image_count = len([img for img in os.listdir(output_class_dir) if img.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Processing class '{soil_class}' with {image_count} images.")
    
    # Step 3: Augment if needed
    if image_count < TARGET_COUNT:
        augment_images(class_dir, output_class_dir, TARGET_COUNT)
        print(f"Augmented '{soil_class}' to {TARGET_COUNT} images.")
    else:
        print(f"'{soil_class}' already balanced.")
