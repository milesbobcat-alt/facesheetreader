from PIL import Image
import pytesseract
import cv2
import numpy as np
from difflib import SequenceMatcher

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
cleaned_image_path = 'cleaned_image.png'
extracted_text = pytesseract.image_to_string(Image.open(cleaned_image_path), config=custom_config, lang='eng')


output_file = "ocr_output.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(extracted_text.strip())
print(f"Success! Text saved to {output_file}")


def normalize(value):
    return ''.join(character.lower() for character in value if character.isalnum())


def find_field_value(words, field_name):
    """Return OCR text on the same line and to the right of a field label."""
    label_words = field_name.split()
    label_size = len(label_words)
    normalized_label = normalize(field_name)
    best_match = None
    best_score = 0

    for index in range(len(words) - label_size + 1):
        candidate = ' '.join(word['text'] for word in words[index:index + label_size])
        score = SequenceMatcher(None, normalize(candidate), normalized_label).ratio()
        if score > best_score:
            best_score = score
            best_match = words[index:index + label_size]

    if best_match is None or best_score < 0.78:
        return ''

    label_end = best_match[-1]
    label_center = label_end['top'] + label_end['height'] / 2
    value_words = [
        word for word in words
        if word['left'] > label_end['left'] + label_end['width']
        and word['block_num'] == label_end['block_num']
        and word['par_num'] == label_end['par_num']
        and word['line_num'] == label_end['line_num']
        and abs((word['top'] + word['height'] / 2) - label_center) <= max(label_end['height'], 20)
    ]
    value_words.sort(key=lambda word: word['left'])
    return ' '.join(word['text'] for word in value_words[:8]).strip()


def organize_form(image_path, output_path):
    fields = {
        'Encounter': [
            'Patient Class', 'MRN', 'HAR', 'Patient Type', 'CSN',
            'Hospital Service', 'Unit', 'Admitting Provider', 'Room Bed',
            'Adm Diagnosis', 'Admit Source'
        ],
        'Patient': [
            'Name', 'Address', 'City', 'Race', 'Ethnicity', 'DOB', 'Sex',
            'Language', 'MS', 'Religion', 'Primary Phone'
        ],
        'Guarantor': [
            'Guarantor', 'Address', 'Date of Birth', 'Relation', 'Sex',
            'Guarantor', 'Home Phone', 'Guarantor Employer', 'Work Phone', 'State'
        ],
        'Emergency Contact': [
            'Contact Name', 'Local Contact', 'Relationship & Patient',
            'Primary Phone'
        ],
        'Coverage 1': [
            'Insurance Company', 'Payor Name', 'EFF From - EFF To',
            'Group Number', 'Policy', 'Subscriber Name', 'Subscriber ID',
            'Plan', 'Claim Address', 'Insurance Type', 'Sub. DOB'
        ],
        'Coverage 2': [
            'Insurance Company', 'Payor Name', 'EFF From - EFF To',
            'Group Number', 'Policy', 'Subscriber Name', 'Subscriber ID',
            'Plan', 'Claim Address', 'Insurance Type', 'Sub. DOB'
        ],
        'Coverage 3': [
            'Insurance Company', 'Payor Name', 'EFF From - EFF To',
            'Group Number', 'Policy', 'Subscriber Name', 'Subscriber ID',
            'Plan', 'Claim Address', 'Insurance Type', 'Sub. DOB'
        ]
    }

    data = pytesseract.image_to_data(
        Image.open(image_path), config=custom_config, lang='eng',
        output_type=pytesseract.Output.DICT
    )
    words = [
        {
            'text': data['text'][index].strip(),
            'left': data['left'][index],
            'top': data['top'][index],
            'width': data['width'][index],
            'height': data['height'][index],
            'block_num': data['block_num'][index],
            'par_num': data['par_num'][index],
            'line_num': data['line_num'][index]
        }
        for index in range(len(data['text']))
        if data['text'][index].strip()
    ]

    with open(output_path, 'w', encoding='utf-8') as file:
        for section, section_fields in fields.items():
            file.write(f'[{section}]\n')
            for field in section_fields:
                value = find_field_value(words, field)
                file.write(f'{field}: {value}\n')
            file.write('\n')


organized_output_file = 'organized_output.txt'
organize_form(cleaned_image_path, organized_output_file)
print(f'Organized fields saved to {organized_output_file}')