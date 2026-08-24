from PIL import Image
import pytesseract
import cv2
import numpy as np
from textblob import TextBlob
custom_config = r'--psm 11' #this config should be the best for pytesseract to read a form

#preprocessing with OpenCV

image = cv2.imread("test-face-sheet-1.jpg")
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
denoised = cv2.medianBlur(thresh, 3)
cv2.imwrite("cleaned_image.png", denoised)

#rotating image text (allegedly? not fully sure if I have this right)
coords = np.column_stack(np.where(denoised == 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = denoised.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite("cleaned_image.png", rotated)

#test case just so I could figure out pytesseract
extracted_text = pytesseract.image_to_string(Image.open('cleaned_image.png'), config=custom_config, lang='eng')
corrected_text = extracted_text.strip()
final_text = TextBlob(corrected_text).correct()


output_file = "ocr_output.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(str(final_text))
print(f"Success! Text saved to {output_file}")