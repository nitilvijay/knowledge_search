import os
os.environ["FLAGS_use_onednn"] = "0"

import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm

# -----------------------------
# Initialize OCR
# -----------------------------
print("Initializing PaddleOCR model...")
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# -----------------------------
# Load Image
# -----------------------------
image_path = "News/IMG-20260517-WA0006.jpg"

print(f"Loading image: {image_path}")
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at path: {image_path}")

# -----------------------------
# Preprocessing
# -----------------------------
print("Preprocessing image...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# -----------------------------
# OCR Prediction
# -----------------------------
print("Running OCR text detection and recognition...")
result = ocr.predict(img)

# PaddleOCR returns a list (one entry per image)
texts = result[0]["rec_texts"]
boxes = result[0]["rec_polys"]
scores = result[0]["rec_scores"]

# -----------------------------
# Convert to simple list
# -----------------------------
detections = []

for text, box, score in tqdm(zip(texts, boxes, scores), total=len(texts), desc="Parsing detections"):
    box = np.array(box)

    x = np.min(box[:, 0])
    y = np.min(box[:, 1])

    width = np.max(box[:, 0]) - np.min(box[:, 0])
    height = np.max(box[:, 1]) - np.min(box[:, 1])

    detections.append({
        "text": text,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "score": score,
    })

# -----------------------------
# Sort from top to bottom
# -----------------------------
detections.sort(key=lambda d: d["y"])

# -----------------------------
# Merge into lines
# -----------------------------
lines = []
y_threshold = 15

for det in tqdm(detections, desc="Grouping text into lines"):

    if not lines:
        lines.append([det])
        continue

    last_line = lines[-1]

    avg_y = np.mean([d["y"] for d in last_line])

    if abs(det["y"] - avg_y) < y_threshold:
        last_line.append(det)
    else:
        lines.append([det])

# Sort each line left-to-right
for line in lines:
    line.sort(key=lambda d: d["x"])

# -----------------------------
# Find headline
# -----------------------------
headline = ""
best_score = -1

for line in tqdm(lines, desc="Identifying headline"):

    text = " ".join(d["text"] for d in line)

    avg_height = np.mean([d["height"] for d in line])

    top_y = min(d["y"] for d in line)

    # Larger text near top gets higher score
    score = avg_height - 0.03 * top_y

    if score > best_score:
        best_score = score
        headline = text

# -----------------------------
# Output
# -----------------------------
print("\nHeadline:")
print(headline)

print("\nAll OCR Text:\n")

for line in lines:
    print(" ".join(d["text"] for d in line))