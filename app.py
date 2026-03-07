import os

import gspread
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image as PILImage

from core.analyzer import enrich_with_norms
from core.data_parser import parse_ocr_results
from core.ocr_utils import ocr_image_to_text

# ── Configuration ──────────────────────────────────────────────────────────────
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CREDENTIALS_FILE = "credentials.json"
SHEET_NAME = "historia_badan"
SHEET_COLUMNS = ["Data", "Badanie", "Wynik", "Jednostka", "Min", "Max", "Status"]
CHARTS_DIR = "data/charts"
REPORT_PATH = "data/raport.txt"


# ══════════════════════════════════════════════════════════════════════════════
# Google Sheets helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def connect_to_google_sheets() -> gspread.Worksheet:
    """
    Authenticate with a service account and return the first worksheet of
    the 'historia_badan' spreadsheet.  The connection is cached for the
    lifetime of the Streamlit session.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(
            f"Nie znaleziono pliku '{CREDENTIALS_FILE}'. "
            "Umiec go w glownym katalogu aplikacji."
        )
    gc = gspread.service_account(filename=CREDENTIALS_FILE)
    return gc.open(SHEET_NAME).sheet1


def load_results_from_sheet() -> pd.DataFrame:
    """
    Fetch all rows from Google Sheets and return a typed DataFrame.
    Returns an empty DataFrame (with correct columns) when the sheet is empty.
    """
    sheet = connect_to_google_sheets()
    records = sheet.get_all_records()

    if not records:
        return pd.DataFrame(columns=SHEET_COLUMNS)

    df = pd.DataFrame(records)

    # Coerce numeric columns so that comparisons and plots work correctly
    for col in ("Wynik", "Min", "Max"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return df


def save_results_to_sheet(df_enriched: pd.DataFrame, date: str) -> None:
    """
    Convert the enriched OCR DataFrame (wide format, one column per date) into
    the long format expected by Google Sheets and write it to the sheet.

    Any rows already saved for *date* are replaced (idempotent re-analysis).

    Args:
        df_enriched: DataFrame produced by enrich_with_norms(), containing
                     columns 'Wynik {date}', 'Status {date}', 'Min', 'Max',
                     'Badanie', 'Jedn.'.
        date:        Exam date string in YYYY-MM-DD format.
    """
    wynik_col = f"Wynik {date}"
    status_col = f"Status {date}"

    def _safe_float(value) -> float | str:
        try:
            v = float(str(value).replace(",", "."))
            return "" if pd.isna(v) else v
        except (ValueError, TypeError):
            return ""

    # Build new rows for this date
    new_rows = [
        [
            date,
            row["Badanie"],
            _safe_float(row.get(wynik_col, "")),
            row.get("Jedn.", ""),
            _safe_float(row.get("Min", "")),
            _safe_float(row.get("Max", "")),
            row.get(status_col, ""),
        ]
        for _, row in df_enriched.iterrows()
    ]

    # Load existing data, drop rows for this date (overwrite semantics)
    df_existing = load_results_from_sheet()
    if not df_existing.empty:
        df_existing = df_existing[df_existing["Data"].astype(str) != str(date)]

    df_combined = pd.concat(
        [df_existing, pd.DataFrame(new_rows, columns=SHEET_COLUMNS)],
        ignore_index=True,
    )

    # Rewrite the entire sheet: header row + all data rows
    sheet = connect_to_google_sheets()
    sheet.clear()
    sheet.append_row(SHEET_COLUMNS)
    if not df_combined.empty:
        sheet.append_rows(df_combined.fillna("").values.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# Chart and report generation (operate on the long-format DataFrame)
# ══════════════════════════════════════════════════════════════════════════════

def generate_charts(df_long: pd.DataFrame, charts_dir: str = CHARTS_DIR) -> list[str]:
    """
    For every unique test name, create a trend chart (value over time) and
    save it as a PNG.  Reference range lines are drawn when Min/Max are available.

    Returns a sorted list of file paths to the generated PNG files.
    """
    os.makedirs(charts_dir, exist_ok=True)
    chart_files = []

    for badanie, group in df_long.groupby("Badanie"):
        group = group.sort_values("Data")
        dates = group["Data"].astype(str).tolist()
        values = pd.to_numeric(
            group["Wynik"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).tolist()

        # Use the most recent reference range for the chart lines
        min_series = pd.to_numeric(group["Min"], errors="coerce").dropna()
        max_series = pd.to_numeric(group["Max"], errors="coerce").dropna()
        jednostka = group["Jednostka"].iloc[-1] if "Jednostka" in group.columns else ""

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(dates, values, marker="o", linestyle="-")

        if not min_series.empty:
            ax.axhline(min_series.iloc[-1], color="gray", linestyle="--", linewidth=1)
        if not max_series.empty:
            ax.axhline(max_series.iloc[-1], color="gray", linestyle="--", linewidth=1)

        ax.set_title(badanie)
        ax.set_xlabel("Data badania")
        ax.set_ylabel(jednostka)
        plt.tight_layout()

        # Sanitise test name for use as a filename
        safe_name = "".join(
            c if c.isalnum() or c in " ._-" else "_" for c in str(badanie)
        )
        path = os.path.join(charts_dir, f"{safe_name}.png")
        fig.savefig(path)
        plt.close(fig)
        chart_files.append(path)

    return sorted(chart_files)


def generate_report_text(df_long: pd.DataFrame, output_path: str = REPORT_PATH) -> str:
    """
    Write a plain-text summary report grouped by test name and return its content.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = ["=== RAPORT WYNIKOW BADAN ===\n"]

    for badanie, group in df_long.groupby("Badanie"):
        group = group.sort_values("Data")
        min_series = pd.to_numeric(group["Min"], errors="coerce").dropna()
        max_series = pd.to_numeric(group["Max"], errors="coerce").dropna()
        ref = (
            f"{min_series.iloc[-1]} – {max_series.iloc[-1]}"
            if not min_series.empty and not max_series.empty
            else "brak"
        )
        lines.append(f"Nazwa badania: {badanie}")
        lines.append(f"Zakres referencyjny: {ref}")
        for _, row in group.iterrows():
            lines.append(f"  Wynik {row['Data']}: {row['Wynik']} → {row['Status']}")
        lines.append("")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return content


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Analizator badan laboratoryjnych", layout="wide")
st.title("Analizator badan laboratoryjnych")
st.caption("Wgraj zdjecie z wynikami, a aplikacja odczyta dane i porownna z normami.")

tab_upload, tab_history = st.tabs(["Dodaj badanie", "Historia wynikow"])


# ── Tab 1: Upload & analyse ────────────────────────────────────────────────────
with tab_upload:
    col_img, col_ctrl = st.columns([1, 1], gap="large")

    with col_img:
        uploaded_file = st.file_uploader(
            "Wgraj zdjecie badania (PNG, JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Podglad", use_container_width=True)

    with col_ctrl:
        st.subheader("Ustawienia analizy")
        date = st.date_input(
            "Data badania", value=pd.Timestamp.today()
        ).strftime("%Y-%m-%d")

        analyze_btn = st.button(
            "Analizuj badanie",
            disabled=uploaded_file is None,
            type="primary",
            use_container_width=True,
        )
        if not uploaded_file:
            st.info("Najpierw wgraj zdjecie badania.")

    if uploaded_file and analyze_btn:

        # ── Step 1: OCR ───────────────────────────────────────────────────────
        with st.spinner("Przetwarzanie obrazu przez OCR..."):
            os.makedirs("data", exist_ok=True)
            temp_path = os.path.join("data", "temp_image.png")
            PILImage.open(uploaded_file).save(temp_path)

            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            text = ocr_image_to_text(temp_path)
            df_raw = parse_ocr_results(text)

        with st.expander("Surowy tekst OCR (diagnostyka)"):
            st.text(text)

        if df_raw.empty:
            st.error(
                "Nie udalo sie odczytac wynikow z obrazu. "
                "Sprawdz jakosc i orientacje zdjecia."
            )
            st.stop()

        # ── Step 2: Enrich with norms ─────────────────────────────────────────
        df_enriched = enrich_with_norms(df_raw, date)

        # ── Step 3: Save to Google Sheets ─────────────────────────────────────
        with st.spinner("Zapisywanie wynikow do Google Sheets..."):
            try:
                save_results_to_sheet(df_enriched, date)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(
                    f"Arkusz '{SHEET_NAME}' nie zostal znaleziony w Google Drive. "
                    "Upewnij sie, ze arkusz istnieje i konto serwisowe ma do niego dostep."
                )
                st.stop()
            except gspread.exceptions.APIError as exc:
                st.error(f"Blad API Google Sheets: {exc}")
                st.stop()

        st.success(f"Wyniki z {date} zostaly zapisane w Google Sheets.")

        # ── Step 4: Anomaly summary ───────────────────────────────────────────
        status_col = f"Status {date}"
        wynik_col = f"Wynik {date}"
        if status_col in df_enriched.columns:
            anomalies = df_enriched[
                df_enriched[status_col].isin(["poniżej normy", "powyżej normy"])
            ]
            if anomalies.empty:
                st.success("Wszystkie odczytane wyniki sa w normie.")
            else:
                st.warning(f"{len(anomalies)} wynik(ow) poza norma:")
                for _, r in anomalies.iterrows():
                    direction = (
                        "PONIŻEJ normy"
                        if r[status_col] == "poniżej normy"
                        else "POWYŻEJ normy"
                    )
                    st.markdown(
                        f"- **{r['Badanie']}**: {r[wynik_col]} {r.get('Jedn.', '')} "
                        f"(norma: {r.get('Min', '?')} – {r.get('Max', '?')}) — _{direction}_"
                    )

        # ── Step 5: Results table ─────────────────────────────────────────────
        display_cols = [
            "Badanie", "Jedn.", "Zakres referencyjny", wynik_col, status_col, "Min", "Max",
        ]
        with st.expander("Pelna tabela wynikow z tego badania", expanded=True):
            st.dataframe(
                df_enriched[[c for c in display_cols if c in df_enriched.columns]],
                use_container_width=True,
            )

        # ── Step 6: Charts and report ─────────────────────────────────────────
        with st.spinner("Generowanie wykresow i raportu..."):
            try:
                df_all = load_results_from_sheet()
                chart_files = generate_charts(df_all)
                report_text = generate_report_text(df_all)
            except Exception as exc:
                st.warning(f"Nie udalo sie wygenerowac wykresow: {exc}")
                chart_files = []
                report_text = ""

        if chart_files:
            st.subheader("Wykresy zmian wynikow")
            cols = st.columns(3)
            for i, path in enumerate(chart_files):
                cols[i % 3].image(
                    PILImage.open(path),
                    caption=os.path.splitext(os.path.basename(path))[0],
                    use_container_width=True,
                )

        if report_text:
            col_dl, col_prev = st.columns([1, 2])
            col_dl.download_button(
                "Pobierz raport TXT",
                data=report_text,
                file_name="raport.txt",
                use_container_width=True,
            )
            with col_prev.expander("Podglad raportu"):
                st.text(report_text)


# ── Tab 2: History ─────────────────────────────────────────────────────────────
with tab_history:
    try:
        df_hist = load_results_from_sheet()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(
            f"Arkusz '{SHEET_NAME}' nie zostal znaleziony. "
            "Upewnij sie, ze arkusz istnieje i konto serwisowe ma do niego dostep."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Blad polaczenia z Google Sheets: {exc}")
        st.stop()

    if df_hist.empty:
        st.info(
            "Brak zapisanych wynikow. "
            "Dodaj pierwsze badanie w zakladce 'Dodaj badanie'."
        )
    else:
        dates_available = sorted(df_hist["Data"].astype(str).unique())
        st.caption(f"Zapisane daty badan: {', '.join(dates_available)}")
        st.dataframe(df_hist, use_container_width=True)

        # Show charts generated during the last upload session (if present)
        if os.path.exists(CHARTS_DIR):
            chart_files = sorted(
                f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")
            )
            if chart_files:
                st.subheader("Wykresy trendow")
                cols = st.columns(3)
                for i, file in enumerate(chart_files):
                    cols[i % 3].image(
                        PILImage.open(os.path.join(CHARTS_DIR, file)),
                        caption=os.path.splitext(file)[0],
                        use_container_width=True,
                    )
