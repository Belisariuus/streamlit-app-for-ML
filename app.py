# app.py
"""
Главный файл Streamlit-приложения для анализа данных и построения регрессионных моделей.
Запуск:
    streamlit run app.py
"""
from typing import Optional
import streamlit as st
import pandas as pd
import io
import os
import joblib

from modules.data_loader import load_data_interface
from modules.data_explorer import explore_data_interface
from modules.data_preprocessor import preprocess_interface, save_preprocessing_pipeline
from modules.model_trainer import train_model_interface, save_model_bytes
from modules.metrics_visualizer import visualize_metrics_interface

st.set_page_config(page_title="Data Analysis & Regression Studio", layout="wide")

APP_TITLE = "Data Analysis & Regression Studio"

def main() -> None:
    st.title(APP_TITLE)
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите модуль", (
        "Загрузка данных", "Разведочный анализ", "Предобработка", "Обучение модели", "Визуализация метрик", "Справка/README"
    ))

    # Global session state containers
    if "df" not in st.session_state:
        st.session_state.df = None  # type: ignore
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None  # type: ignore
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None  # type: ignore
    if "model" not in st.session_state:
        st.session_state.model = None  # type: ignore
    if "train_results" not in st.session_state:
        st.session_state.train_results = None  # type: ignore

    try:
        if page == "Загрузка данных":
            df = load_data_interface()
            if df is not None:
                st.session_state.df = df
        elif page == "Разведочный анализ":
            if st.session_state.df is None:
                st.info("Сначала загрузите данные в разделе 'Загрузка данных'.")
            else:
                explore_data_interface(st.session_state.df)
        elif page == "Предобработка":
            if st.session_state.df is None:
                st.info("Сначала загрузите данные в разделе 'Загрузка данных'.")
            else:
                processed_df, pipeline = preprocess_interface(st.session_state.df)
                if processed_df is not None:
                    st.session_state.processed_df = processed_df
                    st.session_state.pipeline = pipeline
                    # Offer download
                    buf = io.BytesIO()
                    processed_df.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button("Скачать обработанные данные (CSV)", data=buf, file_name="processed_data.csv")
                    # Save pipeline
                    if pipeline is not None:
                        save_preprocessing_pipeline(pipeline, "preprocessing_pipeline.joblib")
                        with open("preprocessing_pipeline.joblib", "rb") as f:
                            st.download_button("Скачать pipeline предобработки", data=f, file_name="preprocessing_pipeline.joblib")
        elif page == "Обучение модели":
            if st.session_state.processed_df is None:
                st.info("Сначала выполните предобработку данных в разделе 'Предобработка'.")
            else:
                model, results = train_model_interface(st.session_state.processed_df)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.train_results = results
                    # Allow model download (joblib bytes)
                    model_bytes = save_model_bytes(model)
                    st.download_button("Скачать обученную модель (joblib)", data=model_bytes, file_name="trained_model.joblib")
        elif page == "Визуализация метрик":
            if st.session_state.train_results is None:
                st.info("Сначала обучите модель в разделе 'Обучение модели'.")
            else:
                visualize_metrics_interface(st.session_state.train_results)
        else:
            # README / помощь
            with open("README.md", "r", encoding="utf-8") as f:
                readme = f.read()
            st.markdown(readme)
    except Exception as e:
        st.error(f"Непредвиденная ошибка в приложении: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
