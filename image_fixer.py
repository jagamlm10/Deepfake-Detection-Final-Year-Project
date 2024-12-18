import os
from PIL import Image

def fix_png_images(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    # Check if the image is in RGB mode, if not, convert it
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Remove ICC profile by saving without it
                    img.save(file_path, 'PNG')
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Replace 'path/to/folder' with your actual folder path
fix_png_images("./art_dataset/Train/Real Images")
