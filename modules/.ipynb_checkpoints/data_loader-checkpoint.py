# modules/data_loader.py
"""
Модуль 1: Загрузка данных
Интерфейс для загрузки CSV/Excel с подробной обработкой ошибок.
"""
from typing import Optional, Tuple, List
import streamlit as st
import pandas as pd
import chardet
import io

@st.cache_data(show_spinner=False)
def detect_encoding(sample_bytes: bytes) -> str:
    """Определить кодировку по sample_bytes с помощью chardet."""
    try:
        result = chardet.detect(sample_bytes)
        enc = result.get("encoding") or "utf-8"
        return enc
    except Exception:
        return "utf-8"

def _read_csv(file: io.BytesIO, sep: str, encoding: str, header_row: Optional[int], skiprows: Optional[List[int]]) -> pd.DataFrame:
    header = header_row if header_row is not None else "infer"
    df = pd.read_csv(file, sep=sep, encoding=encoding, header=header, skiprows=skiprows)
    return df

def _read_excel(file: io.BytesIO, sheet_name: str, header_row: Optional[int]) -> pd.DataFrame:
    header = header_row if header_row is not None else 0
    df = pd.read_excel(file, sheet_name=sheet_name, header=header, engine="openpyxl")
    return df

def load_data_interface() -> Optional[pd.DataFrame]:
    """
    Streamlit UI: загрузка данных (CSV/Excel), детекция кодировки,
    предпросмотр и базовая проверка.
    """
    st.header("1. Загрузка данных")
    uploaded = st.file_uploader("Выберите файл CSV или Excel", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

    if uploaded is None:
        st.info("Файл не выбран. Загрузите CSV или Excel.")
        return None

    file_bytes = uploaded.read()
    if not file_bytes:
        st.error("Файл пуст.")
        return None

    # Offer preview in guessed encodings
    if uploaded.name.lower().endswith(".csv"):
        st.subheader("Параметры CSV")
        # Try detect
        sample = file_bytes[:10000]
        guessed = detect_encoding(sample)
        st.write(f"Авто-определенная кодировка: `{guessed}`")
        encoding = st.selectbox("Кодировка", options=[guessed, "utf-8", "cp1251", "windows-1251", "latin1"], index=0, help="Попробуйте разные кодировки если есть ошибки")
        sep_choice = st.selectbox("Разделитель (sep)", options=[",", ";", "\t", "|", "auto"], index=0)
        if sep_choice == "auto":
            # Try pandas sniff (simple)
            try:
                import csv
                sniff = csv.Sniffer()
                sample_text = file_bytes[:20000].decode(encoding, errors="replace")
                dialect = sniff.sniff(sample_text)
                sep = dialect.delimiter
            except Exception:
                sep = ","
        else:
            sep = sep_choice

        # Header row: allow selecting which row contains column names
        header_opt = st.radio("Номер строки заголовка", ("Нет", "Первая (0)", "Вторая (1)"), index=1)
        header_row = None
        if header_opt == "Нет":
            header_row = None
        elif header_opt == "Первая (0)":
            header_row = 0
        else:
            header_row = 1

        skip_start = st.number_input("Пропустить начальных строк", min_value=0, max_value=1000, value=0)
        skip_end = st.number_input("Пропустить конечных строк (будут откинуты после чтения)", min_value=0, max_value=1000, value=0)

        try:
            bio = io.BytesIO(file_bytes)
            df = _read_csv(bio, sep=sep, encoding=encoding, header_row=header_row, skiprows=list(range(skip_start)) if skip_start>0 else None)
            if skip_end > 0:
                df = df[:-skip_end]
        except UnicodeDecodeError as e:
            st.error(f"Ошибка декодирования: {e}. Показываю предпросмотр в нескольких кодировках:")
            previews = []
            for enc in ["utf-8", "cp1251", "latin1", "windows-1251"]:
                try:
                    text = file_bytes.decode(enc, errors="replace")
                    previews.append((enc, text[:1000]))
                except Exception:
                    previews.append((enc, "<ошибка при декодировании>"))
            for enc, txt in previews:
                st.markdown(f"**Кодировка {enc}**")
                st.text(txt)
            return None
        except pd.errors.EmptyDataError:
            st.error("Файл пуст или имеет неверный формат CSV.")
            return None
        except Exception as e:
            st.error(f"Ошибка при чтении CSV: {e}")
            return None

    else:
        # Excel
        st.subheader("Параметры Excel")
        try:
            import openpyxl  # ensure installed
        except Exception:
            st.warning("openpyxl не установлен — попытка чтения Excel может не удаться. Установите зависимости.")
        try:
            bio = io.BytesIO(file_bytes)
            # List sheets
            xls = pd.ExcelFile(bio)
            sheets = xls.sheet_names
            sheet = st.selectbox("Выберите лист", options=sheets)
            header_opt = st.radio("Номер строки заголовка", ("Первая (0)", "Вторая (1)", "Нет"), index=0)
            header_row = 0 if header_opt.startswith("Первая") else (1 if header_opt.startswith("Вторая") else None)
            try:
                bio.seek(0)
                df = _read_excel(bio, sheet_name=sheet, header_row=header_row)
            except Exception as e:
                st.error(f"Ошибка при чтении Excel: {e}")
                return None
        except Exception as e:
            st.error(f"Ошибка анализа Excel файла: {e}")
            return None

    # Post-load checks
    try:
        st.write("Предпросмотр данных (первые 10 строк)")
        st.dataframe(df.head(10))
        st.write("Информация о данных")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)
        # Check empty cols/rows
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            st.warning(f"Пустые колонки: {empty_cols}")
        empty_rows_count = (df.isna().all(axis=1)).sum()
        if empty_rows_count > 0:
            st.info(f"Пустых строк (все значения NaN): {empty_rows_count}")
        # Inconsistent columns
        # simple check: CSV with uneven columns would raise earlier; still we check row lengths
        try:
            # ensure columns are unique
            if df.columns.duplicated().any():
                st.warning("Найдены дубликаты имён столбцов.")
        except Exception:
            pass
        # Large file check
        if df.memory_usage(deep=True).sum() > 500 * 1024 * 1024:  # 500 MB
            st.warning("Данные большие (>500MB). Рекомендуется работать с sample или увеличить память.")
            if st.button("Загрузить sample (первые 10000 строк)"):
                df = df.head(10000)
                st.success("Загружен sample.")
                st.dataframe(df.head(5))
        return df
    except Exception as e:
        st.error(f"Ошибка после загрузки: {e}")
        return None
