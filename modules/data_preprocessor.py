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
from sklearn.linear_model import LassoCV, RidgeCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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

def save_preprocessing_pipeline(pipeline: Pipeline, path: str = "preprocessing_pipeline.joblib") -> str:
    """Сохраняет pipeline предобработки и возвращает путь к файлу"""
    try:
        joblib.dump(pipeline, path)
        return path
    except Exception as e:
        st.error(f"Ошибка при сохранении pipeline: {e}")
        return path

def preprocess_interface(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Pipeline]]:
    st.header("3. Предобработка данных")

    try:
        cols = df.columns.tolist()
        st.write("Колонки в датасете:")
        st.dataframe(pd.DataFrame({"columns": cols}))

        # Target selection
        target = st.selectbox("Выберите целевую переменную (для регрессии)", options=["(нет)"] + cols, index=0)

        # Сохраняем исходный target отдельно
        original_target = None
        if target is not None and target != "(нет)":
            original_target = df[target].copy()
            y = df[target].copy()
            X = df.drop(columns=[target])
            st.write(f"Распределение целевой переменной '{target}':")
            st.write(y.value_counts().sort_index())
        else:
            y = None
            X = df.copy()

        # Column-wise operations
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_candidates = [c for c in X.columns if _is_datetime_series(X[c])]

        st.write(f"**Числовые признаки:** {numeric_cols}")
        st.write(f"**Категориальные признаки:** {cat_cols}")
        st.write(f"**Кандидаты datetime:** {datetime_candidates}")

        # Ask user which columns to treat as datetime
        dt_cols = st.multiselect("Преобразовать в datetime (авто-парсинг)", options=datetime_candidates, default=[])
        for c in dt_cols:
            try:
                X[c] = pd.to_datetime(X[c], errors="coerce", infer_datetime_format=True)
            except Exception:
                st.warning(f"Не удалось автопарсить колонку {c} в datetime.")

        # Missing value strategies
        st.subheader("Обработка пропусков")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Числовые признаки**")
            num_strategy = st.selectbox("Стратегия заполнения", options=["mean", "median", "most_frequent", "constant", "drop"], index=0)
            if num_strategy == "constant":
                constant_num = st.number_input("Числовая константа для заполнения", value=0.0)

        with col2:
            st.write("**Категориальные признаки**")
            cat_strategy = st.selectbox("Стратегия заполнения", options=["most_frequent", "constant", "drop"], index=0)
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

        if outlier_method == "IQR":
            iqr_k = st.slider("IQR * k (для границ)", min_value=1.0, max_value=3.0, value=1.5)
        elif outlier_method == "zscore":
            z_thresh = st.slider("Z-score порог", min_value=2.0, max_value=5.0, value=3.0)

        # Text processing
        st.subheader("Текстовые признаки")
        text_cols = st.multiselect("Колонки для текстовой обработки (TF-IDF/Count)", options=cat_cols)
        text_method = None
        if text_cols:
            text_method = st.selectbox("Метод представления текста", options=["tfidf", "count"], index=0)
            max_features = st.number_input("Макс. фич в векторизаторе", min_value=100, max_value=50000, value=2000)

        # Feature engineering
       # st.subheader("Feature engineering")
       # poly_degree = st.slider("Степень полиномиальных признаков (0 - нет)", 0, 3, 0)
       # date_features = st.multiselect("Извлечь признаки из datetime", options=dt_cols)

        # Балансировка данных
        st.subheader("Балансировка классов")
        balance_method = st.selectbox(
            "Выберите метод балансировки",
            options=["none", "SMOTE", "RandomOverSampler", "RandomUnderSampler"],
            index=0
        )

        # Автоматическая оптимизация признаков
        st.subheader("Автоматическая оптимизация признаков")
        feature_optimization = st.selectbox(
            "Метод оптимизации признаков",
            options=["none", "LassoCV", "RidgeCV"],
            index=0
        )

        if feature_optimization != "none":
            if feature_optimization == "LassoCV":
                n_alphas = st.slider("Количество альфа для LassoCV", 10, 100, 50)
            else:
                n_alphas = st.slider("Количество альфа для RidgeCV", 10, 100, 50)

        # Preview transformations and build pipeline when user clicks
        if st.button("Применить предобработку"):
            with st.spinner("Применяю предобработку..."):
                try:
                    # Копируем данные для обработки
                    X_proc = X.copy()

                    # ИСКЛЮЧАЕМ ТАРГЕТ ИЗ ОБРАБОТКИ ВЫБРОСОВ И МАСШТАБИРОВАНИЯ
                    # Обрабатываем только фичи (X_proc), target остается неизменным

                    # Handle missing values
                    numeric_cols_actual = X_proc.select_dtypes(include=["number"]).columns.tolist()
                    cat_cols_actual = X_proc.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

                    # Numeric imputation
                    if numeric_cols_actual:
                        if num_strategy == "drop":
                            X_proc.drop(columns=[c for c in numeric_cols_actual if X_proc[c].isna().all()], inplace=True)
                        else:
                            for c in numeric_cols_actual:
                                if X_proc[c].isna().any():
                                    if num_strategy == "mean":
                                        X_proc[c] = X_proc[c].fillna(X_proc[c].mean())
                                    elif num_strategy == "median":
                                        X_proc[c] = X_proc[c].fillna(X_proc[c].median())
                                    elif num_strategy == "most_frequent":
                                        X_proc[c] = X_proc[c].fillna(X_proc[c].mode().iloc[0] if not X_proc[c].mode().empty else 0)
                                    elif num_strategy == "constant":
                                        X_proc[c] = X_proc[c].fillna(constant_num)

                    # Categorical imputation
                    if cat_cols_actual:
                        if cat_strategy == "drop":
                            X_proc.drop(columns=[c for c in cat_cols_actual if X_proc[c].isna().all()], inplace=True)
                        else:
                            for c in cat_cols_actual:
                                if X_proc[c].isna().any():
                                    if cat_strategy == "most_frequent":
                                        X_proc[c] = X_proc[c].fillna(X_proc[c].mode().iloc[0] if not X_proc[c].mode().empty else "Unknown")
                                    elif cat_strategy == "constant":
                                        X_proc[c] = X_proc[c].fillna(constant_cat)

                    # Label encode fallback
                    if cat_cols_actual and encode_strategy == "label-encode":
                        for c in cat_cols_actual:
                            X_proc[c], _ = pd.factorize(X_proc[c].astype(str), sort=True)

                    # Text vectorization
                    text_pipelines = {}
                    if text_cols:
                        for tc in text_cols:
                            try:
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

                                texts = vec.fit_transform(X_proc[tc].fillna("").astype(str))
                                cols_text = [f"{tc}_text_{i}" for i in range(texts.shape[1])]
                                df_text = pd.DataFrame(texts.toarray() if hasattr(texts, "toarray") else texts,
                                                       columns=cols_text, index=X_proc.index)
                                X_proc = pd.concat([X_proc.drop(columns=[tc]), df_text], axis=1)
                                text_pipelines[tc] = vec
                            except Exception as e:
                                st.warning(f"Ошибка при векторизации текста {tc}: {e}")

                    # Date features
                    for dcol in date_features:
                        try:
                            X_proc[f"{dcol}_year"] = X_proc[dcol].dt.year
                            X_proc[f"{dcol}_month"] = X_proc[dcol].dt.month
                            X_proc[f"{dcol}_day"] = X_proc[dcol].dt.day
                            X_proc[f"{dcol}_weekday"] = X_proc[dcol].dt.weekday
                        except Exception:
                            st.warning(f"Не удалось извлечь признаки из {dcol}")

                    # Outlier handling - ТОЛЬКО ДЛЯ ФИЧЕЙ, НЕ ДЛЯ ТАРГЕТА
                    numeric_cols_current = X_proc.select_dtypes(include=["number"]).columns.tolist()

                    if outlier_method == "IQR" and numeric_cols_current:
                        for c in numeric_cols_current:
                            q1 = X_proc[c].quantile(0.25)
                            q3 = X_proc[c].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - iqr * iqr_k
                            upper = q3 + iqr * iqr_k
                            X_proc.loc[X_proc[c] < lower, c] = np.nan
                            X_proc.loc[X_proc[c] > upper, c] = np.nan
                            # Reimpute outliers
                            if num_strategy != "drop":
                                X_proc[c] = X_proc[c].fillna(X_proc[c].median())

                    elif outlier_method == "zscore" and numeric_cols_current:
                        for c in numeric_cols_current:
                            col_z = (X_proc[c] - X_proc[c].mean()) / X_proc[c].std(ddof=0)
                            X_proc.loc[col_z.abs() > z_thresh, c] = np.nan
                            if num_strategy != "drop":
                                X_proc[c] = X_proc[c].fillna(X_proc[c].median())

                    elif outlier_method == "winsorize" and numeric_cols_current:
                        from scipy.stats.mstats import winsorize
                        for c in numeric_cols_current:
                            try:
                                arr = X_proc[c].copy().astype(float)
                                lower_p = 0.05
                                upper_p = 0.95
                                arr_w = winsorize(arr, (lower_p, 1-upper_p))
                                X_proc[c] = arr_w
                            except Exception as e:
                                st.warning(f"Не удалось применить winsorize к {c}: {e}")

                    # Scaling - ТОЛЬКО ДЛЯ ФИЧЕЙ
                    if scale_choice != "none" and numeric_cols_current:
                        if scale_choice == "standard":
                            scaler = StandardScaler()
                        elif scale_choice == "minmax":
                            scaler = MinMaxScaler()
                        else:
                            scaler = RobustScaler()
                        X_proc[numeric_cols_current] = scaler.fit_transform(X_proc[numeric_cols_current])

                    # Drop any fully empty cols
                    X_proc.dropna(axis=1, how="all", inplace=True)

                    # Автоматическая оптимизация признаков
                    important_features = None
                    if feature_optimization != "none" and y is not None:
                        try:
                            # Убедимся, что все данные числовые
                            X_for_optimization = X_proc.select_dtypes(include=["number"]).copy()

                            if not X_for_optimization.empty:
                                if feature_optimization == "LassoCV":
                                    model = LassoCV(cv=5, n_alphas=n_alphas, random_state=42)
                                else:
                                    model = RidgeCV(cv=5, alphas=np.logspace(-3, 3, n_alphas))

                                model.fit(X_for_optimization, y)

                                # Выбираем важные признаки
                                if feature_optimization == "LassoCV":
                                    # Для Lasso - признаки с ненулевыми коэффициентами
                                    important_features = X_for_optimization.columns[model.coef_ != 0].tolist()
                                else:
                                    # Для Ridge - признаки с наибольшими абсолютными коэффициентами
                                    coef_abs = np.abs(model.coef_)
                                    threshold = np.percentile(coef_abs, 50)  # Берем верхние 50%
                                    important_features = X_for_optimization.columns[coef_abs >= threshold].tolist()

                                if important_features:
                                    X_proc = X_proc[important_features]
                                    st.success(f"Отобрано {len(important_features)} важных признаков методом {feature_optimization}")
                                else:
                                    st.warning("Метод оптимизации не отобрал важные признаки")

                        except Exception as e:
                            st.warning(f"Ошибка при оптимизации признаков: {e}")

                    # Балансировка данных
                    if y is not None and balance_method != "none":
                        try:
                            # Проверяем, является ли задача классификацией (бинарной или многоклассовой)
                            unique_classes = y.nunique()
                            if unique_classes <= 10:  # Считаем это классификацией
                                if balance_method == "SMOTE":
                                    sampler = SMOTE(random_state=42)
                                elif balance_method == "RandomOverSampler":
                                    sampler = RandomOverSampler(random_state=42)
                                else:
                                    sampler = RandomUnderSampler(random_state=42)

                                X_res, y_res = sampler.fit_resample(X_proc, y)

                                # Создаем новый DataFrame с сбалансированными данными
                                df_balanced = pd.concat([pd.DataFrame(X_res, columns=X_proc.columns),
                                                         pd.Series(y_res, name=target)], axis=1)

                                st.success(f"Балансировка методом {balance_method} выполнена. "
                                           f"Распределение классов: {dict(zip(*np.unique(y_res, return_counts=True)))}")

                                # Обновляем данные
                                X_proc = X_res
                                y = y_res
                            else:
                                st.info("Балансировка применяется только для задач классификации (<=10 уникальных классов)")

                        except Exception as e:
                            st.warning(f"Не удалось применить балансировку: {e}")

                    # Собираем финальный DataFrame
                    if y is not None:
                        df_final = pd.concat([X_proc, y.reset_index(drop=True)], axis=1)
                    else:
                        df_final = X_proc.copy()

                    # Строим pipeline для сохранения
                    pipeline = Pipeline([
                        ('preprocessor', ColumnTransformer(
                            transformers=[
                                ('num', Pipeline([
                                    ('imputer', SimpleImputer(strategy=num_strategy if num_strategy != "constant" else "constant",
                                                              fill_value=constant_num if num_strategy == "constant" else None)),
                                    ('scaler', StandardScaler() if scale_choice == "standard" else
                                    MinMaxScaler() if scale_choice == "minmax" else
                                    RobustScaler() if scale_choice == "robust" else 'passthrough')
                                ]), numeric_cols_actual),
                                ('cat', Pipeline([
                                    ('imputer', SimpleImputer(strategy=cat_strategy if cat_strategy != "constant" else "constant",
                                                              fill_value=constant_cat if cat_strategy == "constant" else None)),
                                    ('encoder', OneHotEncoder(drop='first' if onehot_drop else None) if encode_strategy == "onehot" else
                                    OrdinalEncoder() if encode_strategy == "ordinal" else 'passthrough')
                                ]), cat_cols_actual)
                            ],
                            remainder='passthrough'
                        ))
                    ])

                    # Показываем результаты
                    st.subheader("Результаты предобработки")
                    st.write(f"Размерность данных после обработки: {df_final.shape}")

                    if y is not None:
                        st.write(f"Распределение целевой переменной после обработки:")
                        target_counts = df_final[target].value_counts().sort_index()
                        st.write(target_counts)

                        # Проверяем, не стал ли target нулевым
                        if target_counts.sum() == 0:
                            st.error("ВНИМАНИЕ: Целевая переменная стала полностью нулевой после обработки!")
                            st.info("Рекомендации: проверьте настройки обработки выбросов и убедитесь, что target исключен из обработки")

                    st.write("Первые 5 строк обработанных данных:")
                    st.dataframe(df_final.head())

                    # Сохраняем pipeline
                    try:
                        save_preprocessing_pipeline(pipeline)
                        st.success("Pipeline предобработки сохранен в файл 'preprocessing_pipeline.joblib'")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить pipeline: {e}")

                    return df_final, pipeline

                except Exception as e:
                    st.error(f"Ошибка при применении предобработки: {e}")
                    import traceback
                    st.error(f"Детали ошибки: {traceback.format_exc()}")
                    return None, None

        else:
            st.info("Настройте параметры и нажмите 'Применить предобработку'.")
            return None, None

    except Exception as e:
        st.error(f"Ошибка в интерфейсе предобработки: {e}")
        import traceback
        st.error(f"Детали ошибки: {traceback.format_exc()}")
        return None, None
