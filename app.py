import os
import platform

import gspread
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import streamlit as st
from google.oauth2.service_account import Credentials
from PIL import Image as PILImage

from core.analyzer import enrich_with_norms
from core.data_parser import parse_ocr_results
from core.ocr_utils import ocr_image_to_text

# ── Configuration ──────────────────────────────────────────────────────────────
# On Windows, tesseract is not in PATH so the binary path must be set explicitly.
# On Linux (Streamlit Cloud), tesseract is installed via packages.txt and in PATH.
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    Authenticate via st.secrets["gcp_service_account"] and return the first
    worksheet of the 'historia_badan' spreadsheet.
    Credentials are never stored in the repo — set them in .streamlit/secrets.toml
    locally or in the Streamlit Cloud dashboard for deployment.
    """
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).sheet1


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


def _compute_status(wynik, min_val, max_val) -> str:
    try:
        w = float(wynik)
        mn = float(min_val) if min_val != "" else None
        mx = float(max_val) if max_val != "" else None
        if mn is not None and w < mn:
            return "poniżej normy"
        if mx is not None and w > mx:
            return "powyżej normy"
        return "w normie"
    except (ValueError, TypeError):
        return ""


def save_long_rows_to_sheet(rows: list[dict]) -> None:
    """
    Save manually entered rows (already in long format) to Google Sheets.
    Overwrites existing rows that share the same Data + Badanie combination.

    Each dict must have keys matching SHEET_COLUMNS:
    Data, Badanie, Wynik, Jednostka, Min, Max, Status.
    """
    df_existing = load_results_from_sheet()

    for row in rows:
        key_date = str(row["Data"])
        key_test = str(row["Badanie"])
        if not df_existing.empty:
            df_existing = df_existing[
                ~(
                    (df_existing["Data"].astype(str) == key_date)
                    & (df_existing["Badanie"].astype(str) == key_test)
                )
            ]

    df_new = pd.DataFrame(rows, columns=SHEET_COLUMNS)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

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
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_results_cards(df_enriched: pd.DataFrame, date: str) -> None:
    """Render styled analysis summary and test-result cards."""
    wynik_col = f"Wynik {date}"
    status_col = f"Status {date}"

    if wynik_col not in df_enriched.columns or status_col not in df_enriched.columns:
        st.dataframe(df_enriched, use_container_width=True)
        return

    statuses = df_enriched[status_col].fillna("")
    n_normal = int((statuses == "w normie").sum())
    n_low    = int((statuses == "poniżej normy").sum())
    n_high   = int((statuses == "powyżej normy").sum())
    abnormal_names = df_enriched[
        statuses.isin(["poniżej normy", "powyżej normy"])
    ]["Badanie"].tolist()

    # ── Analysis Summary cards ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="small")
    c1.markdown(f"""
<div style="background:#EAF5EF;border-radius:16px;padding:20px 24px;">
  <div style="color:#2D7A5C;font-weight:600;font-size:14px;">✓&nbsp; W normie</div>
  <div style="font-size:44px;font-weight:700;color:#1A1A1A;margin-top:8px;line-height:1;">{n_normal}</div>
</div>""", unsafe_allow_html=True)
    c2.markdown(f"""
<div style="background:#FDF6EE;border-radius:16px;padding:20px 24px;">
  <div style="color:#B5520F;font-weight:600;font-size:14px;">↗&nbsp; Poniżej normy</div>
  <div style="font-size:44px;font-weight:700;color:#1A1A1A;margin-top:8px;line-height:1;">{n_low}</div>
</div>""", unsafe_allow_html=True)
    c3.markdown(f"""
<div style="background:#FEEDED;border-radius:16px;padding:20px 24px;">
  <div style="color:#C0392B;font-weight:600;font-size:14px;">⚠&nbsp; Powyżej normy</div>
  <div style="font-size:44px;font-weight:700;color:#1A1A1A;margin-top:8px;line-height:1;">{n_high}</div>
</div>""", unsafe_allow_html=True)

    # ── Attention Required box ────────────────────────────────────────────────
    if abnormal_names:
        pills = "".join(
            f'<span style="background:#FFFFFF;border-radius:20px;padding:5px 16px;'
            f'margin:4px 4px 0 0;display:inline-block;font-size:13px;font-weight:500;'
            f'border:1px solid #DDD8CC;">{name}</span>'
            for name in abnormal_names
        )
        st.markdown(f"""
<div style="background:#EDE4D0;border-radius:16px;padding:20px 24px;margin-top:16px;">
  <div style="font-weight:700;font-size:15px;margin-bottom:6px;">ⓘ Wymaga uwagi</div>
  <div style="font-size:14px;color:#5C4A2A;margin-bottom:12px;">
    {len(abnormal_names)} wynik(i) poza normą. Skonsultuj się z lekarzem.
  </div>
  <div>{pills}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Test Result Cards ─────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:22px;font-weight:700;color:#1A1A1A;margin-bottom:4px;'>"
        "Wyniki badań</div>",
        unsafe_allow_html=True,
    )

    rows = df_enriched.to_dict("records")
    cols = st.columns(3, gap="medium")

    for i, row in enumerate(rows):
        name      = str(row.get("Badanie", ""))
        wynik_raw = row.get(wynik_col, "")
        status    = str(row.get(status_col, ""))
        jedn      = str(row.get("Jedn.", ""))
        min_val   = row.get("Min", "")
        max_val   = row.get("Max", "")

        # Badge & color
        if status == "w normie":
            badge_bg, badge_color, badge_label = "#E8F5EF", "#2D7A5C", "W normie"
            bar_color, status_icon = "#2D7A5C", "✓"
        elif status == "powyżej normy":
            badge_bg, badge_color, badge_label = "#FEEAEA", "#C0392B", "Powyżej normy"
            bar_color, status_icon = "#C0392B", "⚠"
        elif status == "poniżej normy":
            badge_bg, badge_color, badge_label = "#FEF3E8", "#B5520F", "Poniżej normy"
            bar_color, status_icon = "#D4713A", "↗"
        else:
            badge_bg, badge_color, badge_label = "#F0F0F0", "#666666", "—"
            bar_color, status_icon = "#CCCCCC", "—"

        # Range bar fill %
        pct = 50.0
        try:
            v  = float(str(wynik_raw).replace(",", "."))
            mn = float(str(min_val)) if str(min_val) not in ("", "nan") else None
            mx = float(str(max_val)) if str(max_val) not in ("", "nan") else None
            if mn is not None and mx is not None and mx > mn:
                pct = min(100.0, max(0.0, (v - mn) / (mx - mn) * 100))
        except (ValueError, TypeError):
            pass

        # Display value
        try:
            v_disp = f"{float(str(wynik_raw).replace(',', '.')):.4g}"
        except (ValueError, TypeError):
            v_disp = str(wynik_raw)

        min_disp = "" if str(min_val) in ("", "nan") else str(min_val)
        max_disp = "" if str(max_val) in ("", "nan") else str(max_val)

        range_html = ""
        if min_disp or max_disp:
            range_html = f"""
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#999;margin-top:14px;">
    <span>{min_disp}</span>
    <span style="color:#BBBBBB;">Zakres normalny</span>
    <span>{max_disp}</span>
  </div>
  <div style="background:#E8E5DF;border-radius:4px;height:6px;margin-top:4px;overflow:hidden;">
    <div style="background:{bar_color};height:6px;width:{pct:.1f}%;border-radius:4px;"></div>
  </div>"""

        card_html = f"""
<div style="background:#FFFFFF;border-radius:16px;padding:20px 22px;margin-bottom:16px;
     box-shadow:0 1px 6px rgba(0,0,0,0.06);border:1px solid #F0EDE8;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-weight:700;font-size:15px;color:#1A1A1A;">{name}</span>
    <span style="background:{badge_bg};color:{badge_color};border-radius:20px;
          padding:3px 12px;font-size:12px;font-weight:600;white-space:nowrap;">
      {status_icon}&nbsp;{badge_label}
    </span>
  </div>
  <div style="height:14px;"></div>
  <div style="font-size:34px;font-weight:700;color:#1A1A1A;line-height:1;">
    {v_disp}&nbsp;<span style="font-size:14px;font-weight:400;color:#999;">{jedn}</span>
  </div>
  {range_html}
</div>"""
        cols[i % 3].markdown(card_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Analizator badan laboratoryjnych", layout="wide")

st.markdown("""
<style>
/* ── Page background ────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] { background-color: #F5F3EF !important; }
section[data-testid="stMain"] { background-color: #F5F3EF !important; }

/* ── Block container ────────────────────────────────── */
.block-container { padding-top: 2rem !important; }

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent !important; border-bottom: none !important; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 14px;
    font-weight: 500;
    color: #666 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #1A1A1A !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Primary button ─────────────────────────────────── */
[data-testid="stBaseButton-primary"] {
    background-color: #2D7A5C !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background-color: #256349 !important;
}
[data-testid="stBaseButton-secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] { border-radius: 12px !important; }

/* ── Expander ───────────────────────────────────────── */
[data-testid="stExpander"] { border-radius: 12px !important; background: #FFFFFF; }

/* ── Headings ───────────────────────────────────────── */
h1 { color: #1A1A1A !important; font-weight: 700 !important; }
h2, h3 { color: #1A1A1A !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Analizator badan laboratoryjnych")
st.caption("Wgraj zdjecie z wynikami, a aplikacja odczyta dane i porownna z normami.")

tab_upload, tab_manual, tab_history = st.tabs(["Dodaj badanie", "Wprowadź ręcznie", "Historia wynikow"])


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
            except KeyError:
                st.error(
                    "Brak sekcji [gcp_service_account] w .streamlit/secrets.toml. "
                    "Dodaj dane konta serwisowego zgodnie z README."
                )
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

        # ── Steps 4+5: Analysis summary + results cards ──────────────────────
        render_results_cards(df_enriched, date)

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


# ── Tab 2: Manual entry ────────────────────────────────────────────────────────
with tab_manual:
    st.subheader("Ręczne wprowadzanie wyników")

    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = []

    # ── Prefill from history ───────────────────────────────────────────────────
    _hist_source = st.session_state.get("hist_df")
    if _hist_source is None:
        try:
            _hist_source = load_results_from_sheet()
        except Exception:
            _hist_source = pd.DataFrame(columns=SHEET_COLUMNS)

    _known_tests = {}
    if not _hist_source.empty:
        for _tname, _grp in _hist_source.groupby("Badanie"):
            _grp_s = _grp.sort_values("Data", ascending=False)
            _mn = pd.to_numeric(_grp_s["Min"], errors="coerce").dropna()
            _mx = pd.to_numeric(_grp_s["Max"], errors="coerce").dropna()
            _known_tests[_tname] = {
                "min": str(_mn.iloc[0]) if not _mn.empty else "",
                "max": str(_mx.iloc[0]) if not _mx.empty else "",
                "unit": str(_grp_s["Jednostka"].iloc[0]) if "Jednostka" in _grp_s else "",
            }

    if _known_tests:
        with st.expander("Uzupełnij normy z historii (opcjonalnie)"):
            _sel = st.selectbox(
                "Wybierz badanie:",
                options=["— wybierz —"] + sorted(_known_tests.keys()),
                key="prefill_select",
            )
            if _sel != "— wybierz —":
                _pf = _known_tests[_sel]
                st.caption(
                    f"Ostatnia norma: **{_pf['min']} – {_pf['max']}**  |  "
                    f"Jednostka: **{_pf['unit']}**"
                )
                if st.button("Zastosuj normy do formularza"):
                    st.session_state["mf_min"] = _pf["min"]
                    st.session_state["mf_max"] = _pf["max"]
                    st.session_state["mf_unit"] = _pf["unit"]
                    st.rerun()

    with st.form("manual_form", clear_on_submit=True):
        col_date, col_name = st.columns([1, 2])
        with col_date:
            manual_date = st.date_input(
                "Data badania", value=pd.Timestamp.today(), key="mf_date"
            ).strftime("%Y-%m-%d")
        with col_name:
            test_name = st.text_input("Nazwa badania (np. Glukoza)", key="mf_name")

        col_val, col_unit, col_min, col_max = st.columns(4)
        with col_val:
            wynik = st.text_input("Wynik", key="mf_wynik")
        with col_unit:
            jednostka = st.text_input("Jednostka (np. mg/dL)", key="mf_unit")
        with col_min:
            ref_min = st.text_input("Min (norma)", key="mf_min")
        with col_max:
            ref_max = st.text_input("Max (norma)", key="mf_max")

        add_btn = st.form_submit_button("Dodaj do listy", use_container_width=True)

    if add_btn:
        if not test_name.strip():
            st.warning("Podaj nazwę badania.")
        elif not wynik.strip():
            st.warning("Podaj wynik.")
        else:
            try:
                wynik_float = float(wynik.replace(",", "."))
                min_float = float(ref_min.replace(",", ".")) if ref_min.strip() else ""
                max_float = float(ref_max.replace(",", ".")) if ref_max.strip() else ""
            except ValueError:
                st.error("Wynik, Min i Max muszą być liczbami.")
                st.stop()

            status = _compute_status(wynik_float, min_float, max_float)
            st.session_state.manual_rows.append(
                {
                    "Data": manual_date,
                    "Badanie": test_name.strip(),
                    "Wynik": wynik_float,
                    "Jednostka": jednostka.strip(),
                    "Min": min_float,
                    "Max": max_float,
                    "Status": status,
                }
            )
            st.success(f"Dodano: {test_name.strip()} = {wynik_float} → {status}")

    if st.session_state.manual_rows:
        st.markdown("**Wyniki do zapisania** (możesz edytować komórki lub usuwać wiersze):")

        df_edit = pd.DataFrame(st.session_state.manual_rows, columns=SHEET_COLUMNS)
        for _col in ("Wynik", "Min", "Max"):
            df_edit[_col] = df_edit[_col].astype(str).replace("nan", "").replace("<NA>", "")

        edited_df = st.data_editor(
            df_edit,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Wynik": st.column_config.TextColumn("Wynik"),
                "Min": st.column_config.TextColumn("Min (norma)"),
                "Max": st.column_config.TextColumn("Max (norma)"),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["w normie", "poniżej normy", "powyżej normy"],
                ),
            },
        )

        # Recompute Status after any edits and sync back to session state
        for _i, _row in edited_df.iterrows():
            edited_df.at[_i, "Status"] = _compute_status(
                _row["Wynik"],
                _row["Min"] if pd.notna(_row["Min"]) else "",
                _row["Max"] if pd.notna(_row["Max"]) else "",
            )
        st.session_state.manual_rows = edited_df.fillna("").to_dict("records")

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("Zapisz do Google Sheets", type="primary", use_container_width=True):
                try:
                    save_long_rows_to_sheet(st.session_state.manual_rows)
                    st.success(f"Zapisano {len(st.session_state.manual_rows)} wynik(ów).")
                    st.session_state.manual_rows = []
                    st.rerun()
                except KeyError:
                    st.error("Brak sekcji [gcp_service_account] w .streamlit/secrets.toml.")
                except gspread.exceptions.SpreadsheetNotFound:
                    st.error(f"Arkusz '{SHEET_NAME}' nie został znaleziony.")
                except gspread.exceptions.APIError as exc:
                    st.error(f"Błąd API Google Sheets: {exc}")
        with col_clear:
            if st.button("Wyczyść listę", use_container_width=True):
                st.session_state.manual_rows = []
                st.rerun()
    else:
        st.info("Lista jest pusta. Dodaj wyniki używając formularza powyżej.")


# ── Tab 3: History ─────────────────────────────────────────────────────────────
with tab_history:
    if "hist_df" not in st.session_state:
        st.session_state.hist_df = None

    # Load from Sheets only on first visit or after an explicit refresh/save
    if st.session_state.hist_df is None:
        try:
            st.session_state.hist_df = load_results_from_sheet()
        except KeyError:
            st.error(
                "Brak sekcji [gcp_service_account] w .streamlit/secrets.toml. "
                "Dodaj dane konta serwisowego zgodnie z README."
            )
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

    df_hist = st.session_state.hist_df

    if df_hist.empty:
        st.info(
            "Brak zapisanych wynikow. "
            "Dodaj pierwsze badanie w zakladce 'Dodaj badanie'."
        )
    else:
        dates_available = sorted(df_hist["Data"].astype(str).unique())
        st.caption(f"Zapisane daty badań: {', '.join(dates_available)}")

        col_refresh, _ = st.columns([1, 3])
        with col_refresh:
            if st.button("Odśwież dane", use_container_width=True):
                st.session_state.hist_df = None
                st.rerun()

        # ── Alphabet filter ───────────────────────────────────────────────────
        all_tests = sorted(df_hist["Badanie"].astype(str).unique())
        first_letters = sorted({t[0].upper() for t in all_tests if t})

        if "alpha_filter" not in st.session_state:
            st.session_state.alpha_filter = "Wszystkie"

        btn_labels = ["Wszystkie"] + first_letters
        btn_cols = st.columns(len(btn_labels))
        for idx, label in enumerate(btn_labels):
            active = st.session_state.alpha_filter == label
            if btn_cols[idx].button(
                label,
                key=f"alpha_{label}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state.alpha_filter = label
                st.rerun()

        _letter = st.session_state.alpha_filter
        if _letter != "Wszystkie":
            df_hist = df_hist[df_hist["Badanie"].astype(str).str.upper().str.startswith(_letter)]

        # ── Pivot table: rows = Badanie, columns = dates ──────────────────────
        df_wynik = df_hist.copy()
        df_wynik["Wynik"] = pd.to_numeric(
            df_wynik["Wynik"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df_status = df_hist[["Data", "Badanie", "Status"]].copy()

        sorted_dates = sorted(df_wynik["Data"].astype(str).unique(), reverse=True)

        pivot = (
            df_wynik.pivot_table(
                index="Badanie", columns="Data", values="Wynik", aggfunc="first"
            )
            .reindex(columns=sorted_dates)
        )
        pivot_status = (
            df_status.pivot_table(
                index="Badanie", columns="Data", values="Status", aggfunc="first"
            )
            .reindex(columns=sorted_dates)
        )

        # Add Min / Max columns from the most recent entry per test
        _norm = (
            df_hist.copy()
            .assign(
                Min=pd.to_numeric(df_hist["Min"].astype(str).str.replace(",", ".", regex=False), errors="coerce"),
                Max=pd.to_numeric(df_hist["Max"].astype(str).str.replace(",", ".", regex=False), errors="coerce"),
            )
            .sort_values("Data", ascending=False)
            .groupby("Badanie")[["Min", "Max"]]
            .first()
        )
        pivot.insert(0, "Max", _norm["Max"])
        pivot.insert(0, "Min", _norm["Min"])

        def _color_cell(val, status):
            if status == "w normie":
                return "background-color: #EAF5EF; color: #2D7A5C;"
            elif status == "powyżej normy":
                return "background-color: #FEEDED; color: #C0392B;"
            elif status == "poniżej normy":
                return "background-color: #FDF6EE; color: #B5520F;"
            return ""

        def _style_pivot(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for row in df.index:
                for col in df.columns:
                    if col in pivot_status.columns and row in pivot_status.index:
                        status = pivot_status.at[row, col]
                        if pd.notna(status):
                            styles.at[row, col] = _color_cell(df.at[row, col], status)
            return styles

        pivot.columns.name = None  # remove column axis name to avoid subset issues

        def _fmt(v):
            try:
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return " "
                return f"{float(v):.4g}"
            except (TypeError, ValueError):
                return " "

        styled = (
            pivot.style
            .apply(_style_pivot, axis=None)
            .format(_fmt)
        )

        st.dataframe(styled, use_container_width=True)

        # ── Summary: min/max/change per test ──────────────────────────────────
        st.markdown(
            "<div style='font-size:20px;font-weight:700;color:#1A1A1A;"
            "margin-top:32px;margin-bottom:12px;'>Podsumowanie</div>",
            unsafe_allow_html=True,
        )

        summary_rows = []
        for badanie, grp in df_wynik.groupby("Badanie"):
            grp_sorted = grp.dropna(subset=["Wynik"]).sort_values("Data")
            if grp_sorted.empty:
                continue
            vals = grp_sorted["Wynik"].tolist()
            dates = grp_sorted["Data"].tolist()
            min_val = min(vals)
            max_val = max(vals)
            latest = vals[-1]
            latest_date = dates[-1]
            if len(vals) >= 2:
                prev = vals[-2]
                prev_date = dates[-2]
                delta = latest - prev
                delta_pct = (delta / prev * 100) if prev != 0 else 0.0
            else:
                prev = prev_date = delta = delta_pct = None

            # unit
            unit = ""
            u_grp = df_hist[df_hist["Badanie"] == badanie]["Jednostka"]
            if not u_grp.empty:
                unit = str(u_grp.iloc[0])

            summary_rows.append({
                "badanie": badanie,
                "min_val": min_val,
                "max_val": max_val,
                "latest": latest,
                "latest_date": latest_date,
                "prev": prev,
                "prev_date": prev_date,
                "delta": delta,
                "delta_pct": delta_pct,
                "unit": unit,
            })

        cols = st.columns(3, gap="medium")
        for i, s in enumerate(summary_rows):
            u = f" {s['unit']}" if s['unit'] else ""

            if s["delta"] is not None:
                arrow = "↑" if s["delta"] > 0 else ("↓" if s["delta"] < 0 else "→")
                chg_color = "#C0392B" if s["delta"] > 0 else ("#2D7A5C" if s["delta"] < 0 else "#888")
                chg_html = (
                    f'<div style="font-size:13px;color:{chg_color};margin-top:8px;font-weight:600;">'
                    f'{arrow} {s["delta"]:+.4g}{u} ({s["delta_pct"]:+.1f}%) '
                    f'<span style="font-weight:400;color:#999;font-size:12px;">'
                    f'vs {s["prev_date"]}</span></div>'
                )
            else:
                chg_html = '<div style="font-size:13px;color:#AAA;margin-top:8px;">brak poprzedniego pomiaru</div>'

            card = f"""
<div style="background:#FFFFFF;border-radius:14px;padding:18px 20px;margin-bottom:14px;
     box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #F0EDE8;">
  <div style="font-weight:700;font-size:14px;color:#1A1A1A;margin-bottom:6px;">{s['badanie']}</div>
  <div style="display:flex;gap:20px;font-size:13px;color:#555;">
    <span>Min: <b>{s['min_val']:.4g}{u}</b></span>
    <span>Max: <b>{s['max_val']:.4g}{u}</b></span>
    <span style="color:#888;">ostatni: <b style="color:#1A1A1A;">{s['latest']:.4g}{u}</b>
      <span style="font-size:11px;color:#BBB;">({s['latest_date']})</span></span>
  </div>
  {chg_html}
</div>"""
            cols[i % 3].markdown(card, unsafe_allow_html=True)

        # ── Editable raw table (collapsed by default) ─────────────────────────
        with st.expander("Edytuj / usuń wiersze"):
            df_hist_edit = df_hist.copy()
            for _col in ("Wynik", "Min", "Max"):
                df_hist_edit[_col] = (
                    pd.to_numeric(df_hist_edit[_col], errors="coerce")
                    .astype(str)
                    .replace("nan", "")
                )

            edited_hist = st.data_editor(
                df_hist_edit,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Wynik": st.column_config.TextColumn("Wynik"),
                    "Min": st.column_config.TextColumn("Min"),
                    "Max": st.column_config.TextColumn("Max"),
                    "Status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["w normie", "poniżej normy", "powyżej normy"],
                    ),
                },
            )

            if st.button("Zapisz zmiany w historii", type="primary", use_container_width=True):
                try:
                    _sheet = connect_to_google_sheets()
                    _sheet.clear()
                    _sheet.append_row(SHEET_COLUMNS)
                    if not edited_hist.empty:
                        for _c in ("Wynik", "Min", "Max"):
                            edited_hist[_c] = (
                                edited_hist[_c].astype(str)
                                .str.replace(",", ".", regex=False)
                                .replace("nan", "")
                            )
                        _sheet.append_rows(edited_hist.fillna("").values.tolist())
                    st.success("Historia została zaktualizowana.")
                    st.session_state.hist_df = None
                    st.rerun()
                except gspread.exceptions.APIError as exc:
                    st.error(f"Błąd API Google Sheets: {exc}")

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
