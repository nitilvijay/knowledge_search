from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
import pytesseract
from tqdm import tqdm


def count_words(image_path: Path) -> int:
    """OCR an image and return the number of words."""
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 1 --psm 6",
        )
    return len(text.split())


def main():
    image_files = list(Path("News").iterdir())

    if not image_files:
        print("No files found.")
        return

    with ProcessPoolExecutor() as executor:
        word_counts = list(
            tqdm(
                executor.map(count_words, image_files),
                total=len(image_files),
                desc="Processing images",
            )
        )

    print(f"Average number of words per image: {sum(word_counts) / len(word_counts):.2f}")
    print(f"Maximum number of words in a single image: {max(word_counts)}")
    print(f"Minimum number of words in a single image: {min(word_counts)}")


if __name__ == "__main__":
    main()