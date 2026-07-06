import os
import re
from PIL import Image
import pytesseract
from elasticsearch import Elasticsearch
from tqdm import tqdm
from datetime import datetime

def get_date_from_filename(filename):
    """Extract date from filename in format like IMG-YYYYMMDD-*.jpg"""
    # Look for YYYYMMDD pattern in the filename
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None

def main():
    # Initialize Elasticsearch client
    es = Elasticsearch("http://localhost:9200")

    # Folder containing images
    news_folder = "News"

    # Supported image extensions
    extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif")

    # List all image files in the news folder
    image_files = []
    for f in os.listdir(news_folder):
        if f.lower().endswith(extensions):
            image_files.append(os.path.join(news_folder, f))

    if not image_files:
        print("No image files found in the News folder.")
        return

    image_files = image_files[5:]
    print(f"Processing {len(image_files)} images (limited for testing)")

    # Process each image with a progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        # Extract date from filename
        filename = os.path.basename(img_path)
        date_part = get_date_from_filename(filename)
        # print(date_part)
        # Fallback to today's date if no date found in filename
        if not date_part:
            date_part = datetime.now().strftime("%Y-%m-%d")
            print(f"No date found in filename {filename}, using today's date: {date_part}")

        # Perform OCR
        try:
            image = Image.open(img_path)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            print(f"OCR failed for {img_path}: {e}")
            text = ""

        # Prepare document
        headline = ""  # Keep headline empty as requested

        document = {
            "headline": headline,
            "body": text.strip(),
            "date": date_part,
            "img_path": img_path
        }

        # Index to Elasticsearch
        try:
            response = es.index(
                index="articles",
                document=document
            )
            # Optional: print response for debugging
            # print(response)
        except Exception as e:
            print(f"Failed to index {img_path}: {e}")

if __name__ == "__main__":
    main()