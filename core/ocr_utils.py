import pytesseract
from PIL import Image
import re
from datetime import datetime

def get_exam_date():
    user_date = input("📅 Podaj datę badania [RRRR-MM-DD] (Enter = dzisiaj): ").strip()
    if user_date:
        try:
            date = datetime.strptime(user_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print("⚠️ Nieprawidłowy format daty! Użyto dzisiejszej daty.")
            date = datetime.today().strftime("%Y-%m-%d")
    else:
        date = datetime.today().strftime("%Y-%m-%d")
    print(f"📆 Użyta data badania: {date}")
    return date


def ocr_image_to_text(image_path, tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    text = pytesseract.image_to_string(Image.open(image_path), lang="pol")
    clean_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
    return clean_text
