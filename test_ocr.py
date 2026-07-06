from PIL import Image
import pytesseract

img = Image.open('News/IMG-20260615-WA0021.jpg')
text = pytesseract.image_to_string(img)
print(text)