import re
import pandas as pd

def parse_ocr_results(clean_text):
    rows = []
    for line in clean_text.splitlines():
        if not line.strip():
            continue
        match = re.match(r'(.+?)\s+([\d,\.]+)\s+([^\d]+)\s+(.+)', line)
        if match:
            badanie, wynik, jedn, zakres = match.groups()
            if re.search(r'\d', wynik) and re.search(r'\d', zakres):
                rows.append([badanie.strip(), jedn.strip(), zakres.strip(), wynik.strip()])
    df = pd.DataFrame(rows, columns=["Badanie", "Jedn.", "Zakres referencyjny", "Wynik"])
    return df
