# modules/data_explorer.py
"""
Модуль 2: Разведочный анализ данных
Визуализации: распределения, пропуски, корреляции, фильтрация.
"""
from typing import Optional
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

def _basic_info(df: pd.DataFrame) -> None:
    st.subheader("Общая информация")
    st.write(f"Размер датасета: {df.shape[0]} строк × {df.shape[1]} столбцов")
    types = df.dtypes.astype(str).value_counts().to_dict()
    st.write("Типы данных (распределение):")
    st.table(pd.Series(types, name="count"))

def _missing_analysis(df: pd.DataFrame) -> None:
    st.subheader("Анализ пропусков")
    miss = df.isna().sum().sort_values(ascending=False)
    st.write("Количество пропусков по столбцам (по убыванию):")
    st.dataframe(miss.reset_index().rename(columns={"index":"column",0:"missing"}))
    # Bar chart
    top = miss[miss>0].head(50)
    if not top.empty:
        fig, ax = plt.subplots()
        top.plot(kind="bar", ax=ax)
        ax.set_ylabel("Количество пропусков")
        st.pyplot(fig)
    # Heatmap of missing (sample if too large)
    if df.shape[0] > 2000:
        df_sample = df.sample(2000, random_state=0)
    else:
        df_sample = df
    fig, ax = plt.subplots(figsize=(10, min(6, 0.02*df_sample.shape[1])))
    sns.heatmap(df_sample.isna().T, cbar=False, ax=ax)
    ax.set_ylabel("Столбцы")
    st.pyplot(fig)

def _distributions(df: pd.DataFrame) -> None:
    st.subheader("Распределения признаков")
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if numeric:
        col = st.selectbox("Выберите числовую колонку для гистограммы/boxplot", options=numeric)
        fig1, ax1 = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax1)
        ax1.set_title(f"Гистограмма: {col}")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[col].dropna(), ax=ax2)
        ax2.set_title(f"Boxplot: {col}")
        st.pyplot(fig2)
    else:
        st.info("Нет числовых признаков для визуализации.")

    if categorical:
        cat = st.selectbox("Выберите категориальную колонку для countplot", options=categorical)
        fig3, ax3 = plt.subplots(figsize=(8,4))
        top_vals = df[cat].value_counts().nlargest(30)
        sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax3)
        ax3.set_title(f"Countplot (top 30): {cat}")
        st.pyplot(fig3)
    else:
        st.info("Нет категориальных признаков для визуализации.")

def _correlation_analysis(df: pd.DataFrame) -> None:
    st.subheader("Анализ корреляций")
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        st.info("Недостаточно числовых признаков для корреляции.")
        return
    corr = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(min(12, corr.shape[0]), min(10, corr.shape[1])))
    sns.heatmap(corr, annot=False, ax=ax, cmap="coolwarm", center=0)
    st.pyplot(fig)

    # Show top correlated pairs
    corr_values = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    stacked = corr_values.unstack().dropna().sort_values(ascending=False)
    top_pairs = stacked.head(10)
    st.write("Топ коррелирующих пар (по абсолютному значению):")
    st.table(top_pairs.reset_index().rename(columns={"level_0":"feature_1","level_1":"feature_2",0:"abs_corr"}))

    # Scatter for top pair
    if not top_pairs.empty:
        f1, f2 = top_pairs.index[0]
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=numeric[f1], y=numeric[f2], ax=ax2)
        ax2.set_title(f"Scatter: {f1} vs {f2}")
        st.pyplot(fig2)

def _data_preview(df: pd.DataFrame) -> None:
    st.subheader("Детальный просмотр данных")
    st.write("Фильтрация и сортировка:")
    cols = df.columns.tolist()
    sort_col = st.selectbox("Сортировать по столбцу (опционально)", options=["(нет)"] + cols, index=0)
    if sort_col != "(нет)":
        ascending = st.checkbox("По возрастанию", value=False)
        try:
            df = df.sort_values(by=sort_col, ascending=ascending)
        except Exception as e:
            st.warning(f"Невозможно сортировать по выбранному столбцу: {e}")

    # Filtering
    filt_col = st.selectbox("Фильтр: выберите столбец", options=["(нет)"] + cols, index=0, key="filt_col")
    if filt_col != "(нет)":
        unique_vals = df[filt_col].dropna().unique().tolist()
        if len(unique_vals) <= 50:
            chosen = st.multiselect("Выберите значения для фильтра", options=unique_vals)
            if chosen:
                df = df[df[filt_col].isin(chosen)]
        else:
            st.info("Слишком много уникальных значений для выбора; используйте текстовый ввод для фильтра.")
            txt = st.text_input("Фильтр (использует contains)", key="filter_text")
            if txt:
                df = df[df[filt_col].astype(str).str.contains(txt, na=False)]

    start = st.number_input("Показать с строки (0-index)", min_value=0, max_value=max(0, df.shape[0]-1), value=0)
    n = st.number_input("Сколько строк показать", min_value=1, max_value=1000, value=10)
    st.dataframe(df.iloc[start:start+n])

def explore_data_interface(df: pd.DataFrame) -> None:
    """Streamlit interface wrapper для разведочного анализа."""
    st.header("2. Разведочный анализ")
    try:
        _basic_info(df)
        _missing_analysis(df)
        _distributions(df)
        _correlation_analysis(df)
        _data_preview(df)
    except Exception as e:
        st.error(f"Ошибка в модуле разведочного анализа: {e}")
        st.exception(e)
