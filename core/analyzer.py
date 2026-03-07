import pandas as pd
import re

def enrich_with_norms(df, date):
    # Wyodrębnij wartości min i max
    df[['Min', 'Max']] = df['Zakres referencyjny'].apply(
        lambda x: pd.Series(
            tuple(map(float, re.findall(r"[\d\.]+", x.replace(',','.'))[:2])) 
            if len(re.findall(r"[\d\.]+", x.replace(',','.'))) >= 2 
            else (None, None)
        )
    )

    # Status + wynik z datą
    def check_status(val, min_val, max_val):
        try:
            v = float(val.replace(',', '.'))
            if pd.isna(min_val) or pd.isna(max_val): return "brak danych"
            if v < min_val: return "poniżej normy"
            if v > max_val: return "powyżej normy"
            return "w normie"
        except:
            return "brak danych"

    df[f"Wynik {date}"] = df["Wynik"]
    df[f"Status {date}"] = df.apply(lambda r: check_status(r["Wynik"], r["Min"], r["Max"]), axis=1)
    df = df.drop(columns=["Wynik"])
    return df
