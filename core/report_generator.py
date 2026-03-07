import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_report_with_charts(excel_path, output_txt="raport.txt", charts_dir="charts"):
    os.makedirs(charts_dir, exist_ok=True)
    df = pd.read_excel(excel_path)
    wynik_cols = [col for col in df.columns if col.startswith("Wynik")]

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=== RAPORT WYNIKÓW BADAŃ ===\n\n")

        for _, row in df.iterrows():
            f.write(f"Nazwa badania: {row['Badanie']}\n")
            f.write(f"Zakres referencyjny: {row.get('Zakres referencyjny', 'brak')}\n")

            dates, values = [], []
            min_val, max_val = row.get("Min"), row.get("Max")

            for col in wynik_cols:
                wynik = row[col]
                date = col.replace("Wynik ", "")
                dates.append(date)
                try:
                    val = float(str(wynik).replace(",", "."))
                    values.append(val)
                except:
                    values.append(None)
                if pd.isna(wynik) or min_val is None or max_val is None:
                    status = "brak danych"
                else:
                    try:
                        if val < min_val:
                            status = "poniżej normy"
                        elif val > max_val:
                            status = "powyżej normy"
                        else:
                            status = "w normie"
                    except:
                        status = "brak danych"
                f.write(f"  {col}: {wynik} → {status}\n")
            f.write("\n")

            # wykres
            plt.figure(figsize=(5,3))
            plt.plot(dates, values, marker="o", linestyle="-")
            plt.axhline(min_val, color="gray", linestyle="--", linewidth=1)
            plt.axhline(max_val, color="gray", linestyle="--", linewidth=1)
            plt.title(row["Badanie"])
            plt.xlabel("Data badania")
            plt.ylabel(row["Jedn."])
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{row['Badanie']}.png"))
            plt.close()

    print(f"✅ Raport zapisany do: {output_txt}")
