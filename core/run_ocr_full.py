from core.ocr_utils import get_exam_date, ocr_image_to_text
from core.data_parser import parse_ocr_results
from core.analyzer import enrich_with_norms
from core.excel_manager import merge_with_existing2, apply_color_formatting
from core.report_generator import generate_report_with_charts
import pytesseract
from PIL import Image

def run_ocr_full(image_path, excel_path, tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    date = get_exam_date()
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    text = ocr_image_to_text(image_path, tesseract_cmd)
    df_new = parse_ocr_results(text)
    if df_new.empty:
        print("⚠️ Nie znaleziono wyników.")
        return

    df_analyzed = enrich_with_norms(df_new, date)
    df_final = merge_with_existing2(df_analyzed, excel_path, date)
    df_final.drop(columns=[f"Status {date}"], errors="ignore", inplace=True)

    df_final.to_excel(excel_path, index=False)
    apply_color_formatting(excel_path)

    generate_report_with_charts(excel_path)

    print(f"✅ Wyniki zapisane i pokolorowane w pliku: {excel_path}")


# przykład użycia:
# run_ocr_full("test3.png", "data/wyniki.xlsx")
