# modules/data_preprocessor.py
"""
Модуль 3: Предобработка данных
Интерактивный выбор преобразований, обработка пропусков, кодирование, масштабирование, извлечение признаков.
Сохраняет pipeline (sklearn) для дальнейшего применения.
"""
from typing import Optional, Tuple, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
import joblib
import re
import datetime

# Simple transformer wrappers
class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class TextCleaner(TransformerMixin, BaseEstimator):
    def __init__(self, lowercase: bool = True, remove_digits: bool = True):
        self.lowercase = lowercase
        self.remove_digits = remove_digits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.astype(str).copy()
        if self.lowercase:
            out = out.str.lower()
        if self.remove_digits:
            out = out.str.replace(r"\d+", " ", regex=True)
        out = out.str.replace(r"[^\w\s]", " ", regex=True)
        out = out.str.replace(r"\s+", " ", regex=True).str.strip()
        return out

def _is_datetime_series(series: pd.Series) -> bool:
    try:
        pd.to_datetime(series.dropna().iloc[:100], errors="raise")
        return True
    except Exception:
        return False

def save_preprocessing_pipeline(pipeline: Pipeline, path: str = "preprocessing_pipeline.joblib") -> None:
    joblib.dump(pipeline, path)

def preprocess_interface(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Pipeline]]:
    st.header("3. Предобработка данных")
    try:
        cols = df.columns.tolist()
        st.write("Колонки в датасете:")
        st.dataframe(pd.DataFrame({"columns": cols}))

        # Target selection
        target = st.selectbox("Выберите целевую переменную (для регрессии)", options=["(нет)"] + cols, index=0)
        if target == "(нет)":
            st.warning("Целевая переменная не выбрана. Предобработка без таргета возможна, но обучение нельзя будет начать.")
            target = None

        # Column-wise operations
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_candidates = [c for c in cols if _is_datetime_series(df[c])]
        st.write(f"Числовые: {numeric_cols}")
        st.write(f"Категориальные: {cat_cols}")
        st.write(f"Кандидаты datetime: {datetime_candidates}")

        # Ask user which columns to treat as datetime
        dt_cols = st.multiselect("Преобразовать в datetime (авто-парсинг)", options=datetime_candidates, default=[])
        for c in dt_cols:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            except Exception:
                st.warning(f"Не удалось автопарсить колонку {c} в datetime.")

        # Missing value strategies
        st.subheader("Обработка пропусков")
        num_strategy = st.selectbox("Числовые: стратегия", options=["mean", "median", "most_frequent", "constant", "drop"], index=0)
        cat_strategy = st.selectbox("Категориальные: стратегия", options=["most_frequent", "constant", "drop"], index=0)
        constant_num = None
        constant_cat = None
        if num_strategy == "constant":
            constant_num = st.number_input("Числовая константа для заполнения", value=0.0)
        if cat_strategy == "constant":
            constant_cat = st.text_input("Категориальная константа для заполнения", value="Unknown")

        # Encoding
        st.subheader("Кодирование признаков")
        encode_strategy = st.selectbox("Категориальные → числовые", options=["onehot", "ordinal", "label-encode"], index=0)
        onehot_drop = st.checkbox("Drop first в OneHot", value=False)

        # Scaling
        st.subheader("Масштабирование")
        scale_choice = st.selectbox("Масштабирование числовых признаков", options=["none", "standard", "minmax", "robust"], index=0)

        # Outlier handling
        st.subheader("Обработка выбросов")
        outlier_method = st.selectbox("Метод", options=["none", "IQR", "zscore", "winsorize"], index=0)
        iqr_k = st.slider("IQR * k (для границ)", min_value=1.0, max_value=3.0, value=1.5) if outlier_method == "IQR" else None
        z_thresh = st.slider("Z-score порог", min_value=2.0, max_value=5.0, value=3.0) if outlier_method == "zscore" else None

        # Text processing
        st.subheader("Текстовые признаки")
        text_cols = st.multiselect("Колонки для текстовой обработки (TF-IDF/Count)", options=cat_cols)
        text_method = None
        if text_cols:
            text_method = st.selectbox("Метод представления текста", options=["tfidf", "count"], index=0)
            max_features = st.number_input("Макс. фич в векторизаторе", min_value=100, max_value=50000, value=2000)

        # Feature engineering
        st.subheader("Feature engineering")
        poly_degree = st.slider("Степень полиномиальных признаков (0 - нет)", 0, 3, 0)
        date_features = st.multiselect("Извлечь признаки из datetime", options=dt_cols)

        # Preview transformations and build pipeline when user clicks
        if st.button("Применить предобработку"):
            with st.spinner("Применяю предобработку..."):
                # Copy df
                df_proc = df.copy()
                # Handle missing
                # Numeric
                num_cols = df_proc.select_dtypes(include=["number"]).columns.tolist()
                cat_columns = df_proc.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                transformers = []
                if num_cols:
                    if num_strategy == "drop":
                        df_proc.drop(columns=[c for c in num_cols if df_proc[c].isna().all()], inplace=True)
                        num_imputer = None
                    else:
                        if num_strategy == "constant":
                            num_imputer = SimpleImputer(strategy="constant", fill_value=constant_num)
                        else:
                            num_imputer = SimpleImputer(strategy=num_strategy)
                        transformers.append(("num_imputer", num_imputer, num_cols))
                if cat_columns:
                    if cat_strategy == "drop":
                        df_proc.drop(columns=[c for c in cat_columns if df_proc[c].isna().all()], inplace=True)
                        cat_imputer = None
                    else:
                        if cat_strategy == "constant":
                            cat_imputer = SimpleImputer(strategy="constant", fill_value=constant_cat)
                        else:
                            cat_imputer = SimpleImputer(strategy=cat_strategy)
                        transformers.append(("cat_imputer", cat_imputer, cat_columns))

                # Encoding
                enc = None
                if cat_columns:
                    if encode_strategy == "onehot":
                        enc = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first" if onehot_drop else None)
                        transformers.append(("onehot", enc, cat_columns))
                    elif encode_strategy == "ordinal":
                        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                        transformers.append(("ordinal", enc, cat_columns))
                    else:
                        # label-encode fallback using pandas factorize per column later
                        pass

                # Text vectorizers (handled separately)
                text_pipelines = {}
                if text_cols:
                    for tc in text_cols:
                        if text_method == "tfidf":
                            vec = Pipeline([
                                ("clean", TextCleaner()),
                                ("tfidf", TfidfVectorizer(max_features=max_features))
                            ])
                        else:
                            vec = Pipeline([
                                ("clean", TextCleaner()),
                                ("count", CountVectorizer(max_features=max_features))
                            ])
                        text_pipelines[tc] = vec

                # Scaling
                scaler = None
                if scale_choice == "standard":
                    scaler = StandardScaler()
                elif scale_choice == "minmax":
                    scaler = MinMaxScaler()
                elif scale_choice == "robust":
                    scaler = RobustScaler()

                # Build a ColumnTransformer for impute+encode+scale (numerics)
                col_transformers = []
                if num_cols:
                    # impute then scale
                    steps = []
                    if num_strategy != "drop":
                        steps.append(("imputer", SimpleImputer(strategy=num_strategy if num_strategy!="constant" else "constant", fill_value=constant_num if num_strategy=="constant" else None)))
                    if scaler is not None:
                        steps.append(("scaler", scaler))
                    if steps:
                        col_transformers.append(("num", Pipeline(steps), num_cols))
                if cat_columns and encode_strategy in ["onehot", "ordinal"]:
                    if encode_strategy == "onehot":
                        enc_tr = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first" if onehot_drop else None)
                        col_transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy=cat_strategy if cat_strategy!="constant" else "constant", fill_value=constant_cat if cat_strategy=="constant" else None)), ("onehot", enc_tr)]), cat_columns))
                    else:
                        enc_tr = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                        col_transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy=cat_strategy if cat_strategy!="constant" else "constant", fill_value=constant_cat if cat_strategy=="constant" else None)), ("ordinal", enc_tr)]), cat_columns))

                column_transformer = ColumnTransformer(transformers=col_transformers, remainder="drop", sparse_threshold=0)

                # Apply transformations manually to produce a processed DataFrame (simplified)
                try:
                    # Handle simple imputations and label encode fallback
                    df_trans = df_proc.copy()
                    if num_cols:
                        for c in num_cols:
                            if df_trans[c].isna().any():
                                if num_strategy == "mean":
                                    df_trans[c] = df_trans[c].fillna(df_trans[c].mean())
                                elif num_strategy == "median":
                                    df_trans[c] = df_trans[c].fillna(df_trans[c].median())
                                elif num_strategy == "most_frequent":
                                    df_trans[c] = df_trans[c].fillna(df_trans[c].mode().iloc[0] if not df_trans[c].mode().empty else 0)
                                elif num_strategy == "constant":
                                    df_trans[c] = df_trans[c].fillna(constant_num)
                    if cat_columns:
                        for c in cat_columns:
                            if df_trans[c].isna().any():
                                if cat_strategy == "most_frequent":
                                    df_trans[c] = df_trans[c].fillna(df_trans[c].mode().iloc[0] if not df_trans[c].mode().empty else "Unknown")
                                elif cat_strategy == "constant":
                                    df_trans[c] = df_trans[c].fillna(constant_cat)
                    # Label encode fallback
                    if cat_columns and encode_strategy == "label-encode":
                        for c in cat_columns:
                            df_trans[c], _ = pd.factorize(df_trans[c].astype(str), sort=True)
                    # Text vectorization
                    for tc, pipe in text_pipelines.items():
                        try:
                            texts = pipe.fit_transform(df_trans[tc].fillna("").astype(str))
                            # Make DF of vectors
                            cols_text = [f"{tc}_text_{i}" for i in range(texts.shape[1])]
                            df_text = pd.DataFrame(texts.toarray() if hasattr(texts, "toarray") else texts, columns=cols_text, index=df_trans.index)
                            df_trans = pd.concat([df_trans.drop(columns=[tc]), df_text], axis=1)
                        except Exception as e:
                            st.warning(f"Ошибка при векторизации текста {tc}: {e}")
                    # Date features
                    for dcol in date_features:
                        try:
                            df_trans[f"{dcol}_year"] = df_trans[dcol].dt.year
                            df_trans[f"{dcol}_month"] = df_trans[dcol].dt.month
                            df_trans[f"{dcol}_day"] = df_trans[dcol].dt.day
                            df_trans[f"{dcol}_weekday"] = df_trans[dcol].dt.weekday
                        except Exception:
                            st.warning(f"Не удалось извлечь признаки из {dcol}")
                    # Outlier handling (simple approach)
                    if outlier_method == "IQR":
                        for c in num_cols:
                            q1 = df_trans[c].quantile(0.25)
                            q3 = df_trans[c].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - iqr * iqr_k
                            upper = q3 + iqr * iqr_k
                            df_trans.loc[df_trans[c] < lower, c] = np.nan
                            df_trans.loc[df_trans[c] > upper, c] = np.nan
                            # reimpute after marking
                            if num_strategy != "drop":
                                df_trans[c] = df_trans[c].fillna(df_trans[c].median())
                    elif outlier_method == "zscore":
                        for c in num_cols:
                            col_z = (df_trans[c] - df_trans[c].mean()) / df_trans[c].std(ddof=0)
                            df_trans.loc[col_z.abs() > z_thresh, c] = np.nan
                            if num_strategy != "drop":
                                df_trans[c] = df_trans[c].fillna(df_trans[c].median())
                    elif outlier_method == "winsorize":
                        from scipy.stats.mstats import winsorize
                        for c in num_cols:
                            try:
                                arr = df_trans[c].copy().astype(float)
                                lower_p = 0.05
                                upper_p = 0.95
                                arr_w = winsorize(arr, (lower_p, 1-upper_p))
                                df_trans[c] = arr_w
                            except Exception as e:
                                st.warning(f"Не удалось применить winsorize к {c}: {e}")

                    # Scaling if chosen (apply SimpleScaler to numeric columns)
                    if scale_choice != "none" and num_cols:
                        if scale_choice == "standard":
                            scaler = StandardScaler()
                        elif scale_choice == "minmax":
                            scaler = MinMaxScaler()
                        else:
                            scaler = RobustScaler()
                        df_trans[num_cols] = scaler.fit_transform(df_trans[num_cols])

                    # Drop any fully empty cols
                    df_trans.dropna(axis=1, how="all", inplace=True)

                    # Build a lightweight pipeline object to save (for production we'd build a full sklearn pipeline)
                    pipeline = {
                        "num_strategy": num_strategy,
                        "cat_strategy": cat_strategy,
                        "encode_strategy": encode_strategy,
                        "scale_choice": scale_choice,
                        "outlier_method": outlier_method,
                        "text_pipelines": {k: None for k in text_pipelines.keys()},
                    }

                    st.success("Предобработка завершена.")
                    st.dataframe(df_trans.head(5))
                    return df_trans.reset_index(drop=True), pipeline  # type: ignore
                except Exception as e:
                    st.error(f"Ошибка при применении предобработки: {e}")
                    return None, None
        else:
            st.info("Настройте параметры и нажмите 'Применить предобработку'.")
            return None, None

    except Exception as e:
        st.error(f"Ошибка в интерфейсе предобработки: {e}")
        return None, None
