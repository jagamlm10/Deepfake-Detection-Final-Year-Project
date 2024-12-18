import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the dataset directory containing images
dataset_dir = './cifake'  # Update this with the correct path to the main dataset folder

# Subfolders for AI-Generated and Real Images
subfolders = ['AI-Generated Images', 'Real Images']

# Paths to save Train, Validation, and Test datasets
output_dir = './cifake'  # Update this with the path where you want to save the split dataset
train_dir = os.path.join(output_dir, 'Train')
val_dir = os.path.join(output_dir, 'Validation')
test_dir = os.path.join(output_dir, 'Test')

# Create directories for train, validation, and test splits and subfolders for each class
for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(split_dir, 'AI-Generated Images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'Real Images'), exist_ok=True)

# Function to split and copy images
def split_and_copy_images(image_paths, class_name, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split the images into Train, Validation, and Test and copy them to respective folders.

    Args:
        image_paths (list): List of image paths.
        class_name (str): The class name ('AI-Generated Images' or 'Real Images').
        split_ratio (tuple): Ratio for Train, Validation, and Test sets.
    """
    train_size, val_size = split_ratio[0], split_ratio[1]
    
    # Split images into Train, Validation, and Test sets
    train_images, temp_images = train_test_split(image_paths, test_size=1 - train_size, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=split_ratio[2]/(split_ratio[1] + split_ratio[2]), random_state=42)
    
    # Copy images to respective directories
    for image in train_images:
        shutil.copy(image, os.path.join(train_dir, class_name))
    for image in val_images:
        shutil.copy(image, os.path.join(val_dir, class_name))
    for image in test_images:
        shutil.copy(image, os.path.join(test_dir, class_name))

# Loop through both subfolders (AI-Generated and Real Images) and apply the split
for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_dir, subfolder)
    
    # Collect all image paths for the current subfolder (class)
    all_images = [os.path.join(subfolder_path, img) for img in os.listdir(subfolder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split and copy images
    split_and_copy_images(all_images, subfolder)

print("Dataset successfully split into Train, Validation, and Test directories!")
